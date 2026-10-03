"""Equivalent 8-connected RLE labeling; shared pixel primitive, no note policy."""
import numpy as np


def connected_components(mask, min_pixels):
    boolean=np.asarray(mask,dtype=bool)
    edges=np.diff(np.pad(boolean.astype(np.int8),((0,0),(1,1))),axis=1)
    rows,starts=np.nonzero(edges==1)
    _,ends=np.nonzero(edges==-1)
    parent=[];aggregates=[];previous=[];current=[];last_row=-2;overlap_start=0

    def find(label):
        while parent[label]!=label:
            parent[label]=parent[parent[label]]
            label=parent[label]
        return label

    for row,start,end in zip(rows.tolist(),starts.tolist(),ends.tolist()):
        if row!=last_row:
            previous=current if row==last_row+1 else []
            current=[];last_row=row;overlap_start=0
        label=None
        # Each row's runs are ordered and disjoint. Retire runs left of the
        # current start once, then visit only the potentially overlapping
        # prefix. End is exclusive, but equal endpoints remain 8-connected:
        # b == start / a == end represents diagonal neighbouring pixels.
        while overlap_start<len(previous) and previous[overlap_start][1]<start:
            overlap_start+=1
        overlap=overlap_start
        while overlap<len(previous) and previous[overlap][0]<=end:
            prior=find(previous[overlap][2])
            if label is None:
                label=prior
            elif label!=prior:
                # Keep the larger physical component's root. Connected runs
                # do not create fresh labels, so a long ribbon cannot replace
                # its root every row or require a second pass over every run.
                if aggregates[label][4]<aggregates[prior][4]:
                    label,prior=prior,label
                parent[prior]=label
                item,other=aggregates[label],aggregates[prior]
                if other[0]<item[0]:item[0]=other[0]
                if other[1]<item[1]:item[1]=other[1]
                if other[2]>item[2]:item[2]=other[2]
                if other[3]>item[3]:item[3]=other[3]
                item[4]+=other[4]
                if other[5]<item[5]:item[5]=other[5]
            overlap+=1
        if label is None:
            label=len(parent);parent.append(label)
            # The final field is the first row-major component encounter, not
            # its current root or bbox left edge. Late merges retain that order.
            aggregates.append([start,row,end,row+1,end-start,label])
        else:
            item=aggregates[label]
            if start<item[0]:item[0]=start
            if end>item[2]:item[2]=end
            item[3]=row+1
            item[4]+=end-start
        current.append((start,end,label))
    # Filtering is intentionally after all unions: even a one-pixel diagonal
    # bridge can join two accepted components. Weighted roots may reorder IDs,
    # so retain the original first-pixel output order explicitly.
    accepted=[item for label,item in enumerate(aggregates)
              if parent[label]==label and item[4]>=min_pixels]
    accepted.sort(key=lambda item:item[5])
    return [((x,y,right-x,bottom-y),count) for x,y,right,bottom,count,_ in accepted]
