"""Equivalent 8-connected RLE labeling; shared pixel primitive, no note policy."""
import numpy as np


def connected_components(mask, min_pixels):
    boolean=np.asarray(mask,dtype=bool)
    edges=np.diff(np.pad(boolean.astype(np.int8),((0,0),(1,1))),axis=1)
    rows,starts=np.nonzero(edges==1)
    _,ends=np.nonzero(edges==-1)
    parent=[];runs=[];previous=[];current=[];last_row=-2;overlap_start=0

    def find(label):
        while parent[label]!=label:
            parent[label]=parent[parent[label]]
            label=parent[label]
        return label

    for row,start,end in zip(rows.tolist(),starts.tolist(),ends.tolist()):
        if row!=last_row:
            previous=current if row==last_row+1 else []
            current=[];last_row=row;overlap_start=0
        label=len(parent);parent.append(label)
        # Each row's runs are ordered and disjoint. Retire runs left of the
        # current start once, then visit only the potentially overlapping
        # prefix. End is exclusive, but equal endpoints remain 8-connected:
        # b == start / a == end represents diagonal neighbouring pixels.
        while overlap_start<len(previous) and previous[overlap_start][1]<start:
            overlap_start+=1
        overlap=overlap_start
        while overlap<len(previous) and previous[overlap][0]<=end:
            prior=previous[overlap][2]
            left,right=find(label),find(prior)
            if left!=right:parent[right]=left
            overlap+=1
        runs.append((row,start,end,label));current.append((start,end,label))
    aggregates={}
    for row,start,end,label in runs:
        root=find(label)
        if root not in aggregates:
            aggregates[root]=[start,row,end,row+1,end-start]
        else:
            item=aggregates[root]
            item[0]=min(item[0],start);item[1]=min(item[1],row)
            item[2]=max(item[2],end);item[3]=max(item[3],row+1);item[4]+=end-start
    return [((x,y,right-x,bottom-y),count) for x,y,right,bottom,count in aggregates.values() if count>=min_pixels]
