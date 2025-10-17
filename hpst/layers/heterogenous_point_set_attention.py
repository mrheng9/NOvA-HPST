from typing import List, Tuple, Union
from collections import OrderedDict

import sys

import torch
from torch import Tensor, nn
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from torch_geometric.nn.unpool import knn_interpolate
from torch_scatter import scatter
from copy import deepcopy
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr, segment_coo
from torch_scatter.composite import scatter_softmax
from torch.utils.checkpoint import checkpoint

# check if can be removed
from torch_cluster import knn_graph, knn

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, batch = points
        # batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        # offset = batch2offset(batch)
        return [coord, feat, batch], cluster

class PointSetAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        num_heads=4,
        pos_dim=3,
        attn_drop_rate=0.0,
        qkv_bias=True,
    ):
        super(PointSetAttention, self).__init__()
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pos_dim = pos_dim

        self.qkv1 = nn.Linear(embed_channels, 2*3*embed_channels, bias=qkv_bias)
        self.qkv2 = nn.Linear(embed_channels, 2*3*embed_channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj1 = nn.Linear(2*embed_channels, embed_channels)
        self.proj2 = nn.Linear(2*embed_channels, embed_channels)

        self.rpe = nn.Linear(pos_dim, embed_channels)

    def calculate_attention(self, key, query, value, graph, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        key = key[graph[1],...] # # (n_neighbors*n, h, c)
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        value = value[graph[1],...] # (n_neighbors*n, h, c)
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        attn = key * query # (n_neighbors*n, h, c)
        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = attn * value # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat
    
    def calculate_attention_rpe(self, key, query, value, graph, coord, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        coord1 = coord[graph[0],...]
        coord2 = coord[graph[1],...]
        relative_position = coord1 - coord2 # (n_neighbors*n, pos_dim)
        rpe = self.rpe(relative_position)
        rpe = rpe.reshape((-1, self.num_heads, self.embed_channels//self.num_heads)) # (n_neighbors*n, h, c)

        key = key[graph[1],...] # # (n_neighbors*n, h, c)
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        value = value[graph[1],...] # (n_neighbors*n, h, c)
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        attn = key * query + rpe # (n_neighbors*n, h, c)
        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = attn * value # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat

    def calculate_attention_rpe_inference(self, key, query, value, graph, coord, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        #key = key[graph[1],...] # # (n_neighbors*n, h, c)
        attn = key[graph[1],...]
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        #attn = key * query # (n_neighbors*n, h, c)
        attn.mul_(query)

        #del key
        del query

        relative_position = coord[graph[0],...]
        coord2 = coord[graph[1],...]
        relative_position.sub_(coord2) # (n_neighbors*n, pos_dim)
        rpe = self.rpe(relative_position)
        rpe = rpe.reshape((-1, self.num_heads, self.embed_channels//self.num_heads)) # (n_neighbors*n, h, c)

        attn.add_(rpe)


        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = value[graph[1],...] # (n_neighbors*n, h, c)
        feat.mul_(attn) # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat

    def calculate_attention_inference(self, key, query, value, graph, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        #key = key[graph[1],...] # # (n_neighbors*n, h, c)
        attn = key[graph[1],...]
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        #attn = key * query # (n_neighbors*n, h, c)
        attn.mul_(query)

        #del key
        del query

        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = value[graph[1],...] # (n_neighbors*n, h, c)
        feat.mul_(attn) # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat


    def forward(self, feat1, coord1, graph1, feat2, coord2, graph2, graph12, graph21):
        qkv1 = self.qkv1(feat1)
        qkv1 = qkv1.reshape(-1, 2*3, self.num_heads, self.embed_channels//self.num_heads) # (N', V*3*C) => (N', V*3, H, C//H)
        qkv1 = qkv1.permute(1, 0, 2, 3) # (N'1, V*3, H, C//H) => (V*3, N'1, H, C//H)
        query11, key11, value11, query12, key12, value12 = qkv1.unbind(dim=0)

        qkv2 = self.qkv2(feat2)
        qkv2 = qkv2.reshape(-1, 2*3, self.num_heads, self.embed_channels//self.num_heads) # (N', V*3*C) => (N', V*3, H, C//H)
        qkv2 = qkv2.permute(1, 0, 2, 3) # (N'2, V*3, H, C//H) => (V*3, N'2, H, C//H)
        query21, key21, value21, query22, key22, value22 = qkv2.unbind(dim=0)

        n1 = feat1.shape[0]
        n2 = feat2.shape[0]
        
        if self.training:
            feat11 = self.calculate_attention_rpe(key11, query11, value11, graph1, coord1, n1)
            feat22 = self.calculate_attention_rpe(key22, query22, value22, graph2, coord2, n2)

            # print("key1", key1.shape, "query1", query1.shape, "value1", value1.shape, "graph1", graph1.shape, "n1", n1, "key2", key2.shape, "query2", query2.shape, "value2", value2.shape, "graph2", graph2.shape, "n2", n2)
            feat12 = self.calculate_attention(key21, query12, value21, graph21, n1)
            feat21 = self.calculate_attention(key12, query21, value12, graph12, n2)
        else:
            feat11 = self.calculate_attention_rpe_inference(key11, query11, value11, graph1, coord1, n1)
            feat22 = self.calculate_attention_rpe_inference(key22, query22, value22, graph2, coord2, n2)

            # print("key1", key1.shape, "query1", query1.shape, "value1", value1.shape, "graph1", graph1.shape, "n1", n1, "key2", key2.shape, "query2", query2.shape, "value2", value2.shape, "graph2", graph2.shape, "n2", n2)
            feat12 = self.calculate_attention_inference(key21, query12, value21, graph21, n1)
            feat21 = self.calculate_attention_inference(key12, query21, value12, graph12, n2)

        feat1 = torch.cat([feat11, feat12], dim=1)
        feat2 = torch.cat([feat22, feat21], dim=1)

        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)

        return feat1, feat2

class SharedPointSetAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        pos_dim=3,
        attn_drop_rate=0.0,
        qkv_bias=True,
    ):
        super(SharedPointSetAttention, self).__init__()
        self.embed_channels = embed_channels
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pos_dim = pos_dim

        self.linear_q1 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k1 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v1 = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.linear_q2 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k2 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v2 = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj1 = nn.Linear(2*embed_channels, embed_channels)
        self.proj2 = nn.Linear(2*embed_channels, embed_channels)

    def calculate_attention(self, key, query, value, graph, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        key = key[graph[1],...] # # (n_neighbors*n, c)
        query = query[graph[0],...] # (n_neighbors*n, c)
        value = value[graph[0],...] # (n_neighbors*n, c)
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        attn = key * query # (n_neighbors*n, c)
        attn = attn.sum(dim=1) # (n_neighbors * n,)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n,)
        attn = attn.unsqueeze(1)
        attn = self.attn_drop(attn)

        feat = attn * value # (n_neighbors*n, 1) * (n_neighbors*n, c) -> (n_neighbors*n, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, c) -> (n, c)

        return feat


    def forward(self, feat1, coord1, graph1, feat2, coord2, graph2, graph12, graph21):
        query1, key1, value1 = (
            self.linear_q1(feat1),
            self.linear_k1(feat1),
            self.linear_v1(feat1),
        )

        query2, key2, value2 = (
            self.linear_q2(feat2),
            self.linear_k2(feat2),
            self.linear_v2(feat2),
        )
        n1 = feat1.shape[0]
        n2 = feat2.shape[0]
        
        feat11 = self.calculate_attention(key1, query1, value1, graph1, n1)
        feat22 = self.calculate_attention(key2, query2, value2, graph2, n2)

        # print("key1", key1.shape, "query1", query1.shape, "value1", value1.shape, "graph1", graph1.shape, "n1", n1, "key2", key2.shape, "query2", query2.shape, "value2", value2.shape, "graph2", graph2.shape, "n2", n2)
        feat12 = self.calculate_attention(key2, query1, value1, graph21, n1)
        feat21 = self.calculate_attention(key1, query2, value2, graph12, n2)

        feat1 = torch.cat([feat11, feat12], dim=1)
        feat2 = torch.cat([feat22, feat21], dim=1)

        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)

        return feat1, feat2

class IndependentPointSetAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        pos_dim=3,
        attn_drop_rate=0.0,
        qkv_bias=True,
    ):
        super(IndependentPointSetAttention, self).__init__()
        self.embed_channels = embed_channels
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pos_dim = pos_dim

        self.linear_q1 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k1 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v1 = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.linear_q2 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k2 = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v2 = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

    def calculate_attention(self, key, query, value, graph, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        key = key[graph[1],...] # # (n_neighbors*n, c)
        query = query[graph[0],...] # (n_neighbors*n, c)
        value = value[graph[0],...] # (n_neighbors*n, c)
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        attn = key * query # (n_neighbors*n, c)
        attn = attn.sum(dim=1) # (n_neighbors * n,)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n,)
        attn = attn.unsqueeze(1)
        attn = self.attn_drop(attn)

        feat = attn * value # (n_neighbors*n, 1) * (n_neighbors*n, c) -> (n_neighbors*n, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, c) -> (n, c)

        return feat


    def forward(self, feat1, coord1, graph1, feat2, coord2, graph2, graph12, graph21):
        query1, key1, value1 = (
            self.linear_q1(feat1),
            self.linear_k1(feat1),
            self.linear_v1(feat1),
        )

        query2, key2, value2 = (
            self.linear_q2(feat2),
            self.linear_k2(feat2),
            self.linear_v2(feat2),
        )
        n1 = feat1.shape[0]
        n2 = feat2.shape[0]
        
        feat1 = self.calculate_attention(key1, query1, value1, graph1, n1)
        feat2 = self.calculate_attention(key2, query2, value2, graph2, n2)


        return feat1, feat2


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(Block, self).__init__()
        self.attn = PointSetAttention(
            embed_channels=embed_channels,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pos_dim=pos_dim,
            num_heads=num_heads
        )
        self.fc11 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc12 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc31 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc32 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm11 = PointBatchNorm(embed_channels)
        self.norm12 = PointBatchNorm(embed_channels)
        self.norm21 = PointBatchNorm(embed_channels)
        self.norm22 = PointBatchNorm(embed_channels)
        self.norm31 = PointBatchNorm(embed_channels)
        self.norm32 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint

    def forward(self, points1, points2, graph1, graph2, graph12, graph21):
        coord1, feat1, batch1 = points1
        coord2, feat2, batch2 = points2

        identity1 = feat1
        identity2 = feat2

        feat1 = self.act(self.norm11(self.fc11(feat1)))
        feat2 = self.act(self.norm12(self.fc12(feat2)))

        feat1, feat2 = (
            self.attn(feat1, coord1, graph1, feat2, coord2, graph2, graph12, graph21)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat1, coord1, graph1, feat2, coord2, graph2, graph12, graph21)
        )
        feat1 = self.act(self.norm21(feat1))
        feat2 = self.act(self.norm22(feat2))

        feat1 = self.norm31(self.fc31(feat1))
        feat2 = self.norm32(self.fc32(feat2))

        feat1 = identity1 + feat1
        feat2 = identity2 + feat2

        feat1 = self.act(feat1)
        feat2 = self.act(feat2)

        return [coord1, feat1, batch1], [coord2, feat2, batch2]

class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(BlockSequence, self).__init__()

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                enable_checkpoint=enable_checkpoint,
                pos_dim=pos_dim,
                num_heads=num_heads
            )
            self.blocks.append(block)

    def forward(self, points1, points2):
        coord1, feat1, batch1 = points1
        coord2, feat2, batch2 = points2
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        # reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)

        graph1 = knn_graph(coord1, k=self.neighbours, batch=batch1, loop=True, flow='target_to_source')
        graph2 = knn_graph(coord2, k=self.neighbours, batch=batch2, loop=True, flow='target_to_source')
        graph12 = knn(coord1[:,[1]], coord2[:,[1]], k=self.neighbours, batch_x=batch1, batch_y=batch2)
        graph21 = knn(coord2[:,[1]], coord1[:,[1]], k=self.neighbours, batch_x=batch2, batch_y=batch1)
        
        # explanation: knn_graph outputs a list of ordered pairs [[source, target]*nk]
        # where k is the number of neighbors and n is the size of x
        # This list is ordered by sources (flow option) which means that the source column
        # looks like the following: 0,0,0,0...,0 (k times) 1,1,1,1...,1 (k times)
        # hence if we reshape the target column to be of shape (-1, k) it will be the reference
        # indexes, i.e. the ith row will have the k nearest neighbors of the ith sample
        for block in self.blocks:
            points1, points2 = block(points1, points2, graph1, graph2, graph12, graph21)
        return points1, points2
    
class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, batch = points
        skip_coord, skip_feat, skip_batch = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            # UNTESTED - pointops interpolation should correspond with pytorch-geometric unpool.knn_interpolate
            feat = knn_interpolate(feat, coord, skip_coord, batch, skip_batch)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_batch]


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=None,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(Encoder, self).__init__()

        self.down1 = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.down2 = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
            pos_dim=pos_dim,
            num_heads=num_heads
        )

    def forward(self, points1, points2):
        points1, cluster1 = self.down1(points1)
        points2, cluster2 = self.down2(points2)
        points1, points2 = self.blocks(points1, points2)
        return points1, cluster1, points2, cluster2
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
        num_heads=4,
        pos_dim=3
    ):
        super(Decoder, self).__init__()

        self.up1 = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.up2 = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
            num_heads=num_heads,
            pos_dim=pos_dim
        )

    def forward(self, points1, skip_points1, cluster1, points2, skip_points2, cluster2):
        points1 = self.up1(points1, skip_points1, cluster1)
        points2 = self.up2(points2, skip_points2, cluster2)
        return self.blocks(points1, points2)

class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj1 = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
            pos_dim=pos_dim,
            num_heads=num_heads
        )

    def forward(self, points1, points2):
        coord1, feat1, batch1 = points1
        coord2, feat2, batch2 = points2
        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)
        return self.blocks([coord1, feat1, batch1], [coord2, feat2, batch2])