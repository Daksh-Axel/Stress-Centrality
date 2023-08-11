#include<bits/stdc++.h>
#include<cuda.h>
#include <sys/time.h>
#define inf 1000000000
using namespace std;



__device__ void recover_path(int* par,int s,int e, int *path, int *bc,int n){
    if(e==s){
        for(int i=1;i<path[0];i++){
            if(path[i]==e) break;
            bc[path[i]]++;
        }
        return;
    }

    for(int v=1;v<=par[e*n];v++){
        int p=par[e*n+v];
        path[++path[0]]=p;
        recover_path(par,s,p,path,bc,n);
        path[0]--;
    }
}



__device__ void STRESS_CENTRALITY(int src,int n, int *adj,int *stress_cen,int uver,int vver){

    int *par = new int[n*(n+1)];
    int *dist= new int[n];
    for(int i=0;i<n;i++){
        dist[i]=inf;
        par[i*n]=0;
    }
    int *q=new int[n];
    int wcnt=0,rcnt=0;
    
    q[wcnt++]=src;
    dist[src]=0;

    while(rcnt<wcnt){
        int u=q[rcnt++];
        for(int ver=1;ver<=adj[u*n]+1;ver++){
            int v;
            if(ver==adj[u*n]+1){
                if(u==uver) v=vver;
                else if(u==vver) v=uver;
                else break;
            }
            else v=adj[u*n+ver];
            if(dist[v]>dist[u]+1){
                dist[v] = dist[u]+1;
                par[v*n] = 0;
                par[v*n + par[v*n]+1] = u; par[v*n]++;
                q[wcnt++] = v;
            }
            else if(dist[v]==dist[u]+1){
                par[v*n + par[v*n]+1] = u; par[v*n]++;
                
            }
        }
    }
    for(int dst=0;dst<n;dst++){
        if(src!=dst && dist[dst]!=inf){
            int *SC = new int[n]; 
            for(int i=0;i<n;i++) SC[i]=0;
            int *path = new int[n];
            path[0]=0;
            recover_path(par,src,dst,path,SC,n);
            for(int v=0;v<n;v++){
                stress_cen[v]+=SC[v];
            }
            delete[] SC;
            delete[] path;
        }
    }
    delete[] par;
    delete[] dist;
    delete[] q;
    
}
__global__ void SC_kernel(int n,int *adj,int *SC,int u,int v,int *min_SC,int *opt_edge){
    int src = blockIdx.x*blockDim.x + threadIdx.x;
    int *temp = new int[n];
    
    for(int node=0;node<n;node++){
        temp[node]=0;
        SC[node]=0;
    }
    STRESS_CENTRALITY(src,n,adj,temp,u,v);
     __syncthreads();
   
    for(int node = 0;node<n;node++){
        atomicAdd(&SC[node], temp[node]);
    }
     __syncthreads();
    if(src==0){
        for(int node=0;node<n;node++){
          SC[node]/=2;
          if(min_SC[node]>SC[node]){
              min_SC[node]=SC[node];
              opt_edge[node]=u*n+v;
          }
        }
    }
    delete[] temp;
}

int main()
{
    freopen("test.edgelist","r",stdin);
    freopen("op.txt","w+",stdout);
    
    int n,m;
    cin>>n>>m;
    int adj[n*(n+1)];
    int SC[n];
    set<pair<int,int>> edges;
    
    for(int i=0;i<n;i++){
        adj[i*n]=0;
        SC[i]=0;
    }
    for(int i=0;i<m;i++){
        int u,v;
        cin>>u>>v;
        adj[u*n+adj[u*n]+1]=v; adj[u*n]++;
        adj[v*n+adj[v*n]+1]=u; adj[v*n]++;
        edges.insert({u,v});
    }
 
    int *d_adj;
    cudaMalloc(&d_adj,n*(n+1)*sizeof(int));
    cudaMemcpy(d_adj,adj,n*(n+1)*sizeof(int),cudaMemcpyHostToDevice);
 
    int *d_SC;
    cudaMalloc(&d_SC,n*sizeof(int));
    cudaMemcpy(d_SC,SC,n*sizeof(int),cudaMemcpyHostToDevice);
 
    int max_SC[n];for(int i=0;i<n;i++) max_SC[i] = inf;
 
    int *d_min_SC;
    cudaMalloc(&d_min_SC,n*sizeof(int));
    cudaMemcpy(d_min_SC,max_SC,n*sizeof(int),cudaMemcpyHostToDevice);
    
    int *opt_edge;
    cudaMalloc(&opt_edge,n*sizeof(int));
    cudaMemcpy(opt_edge,SC,n*sizeof(int),cudaMemcpyHostToDevice);
    
    SC_kernel<<<1, n>>>(n,d_adj,d_SC,-1,-1,d_min_SC,opt_edge);
    cudaDeviceSynchronize();
    cudaMemcpy(SC,d_min_SC,n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("Initial Stress Centrality:\n");
    for(int i=0;i<n;i++) printf("%d ",SC[i]);
    printf("\n");

 
    for(int u=0;u<n;u++){
        for(int v=u+1;v<n;v++){
            if(edges.find({u,v})!=edges.end()) continue;
            SC_kernel<<<1, n>>>(n,d_adj,d_SC,u,v,d_min_SC,opt_edge);
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                const char* errorMessage = cudaGetErrorString(cudaError);
                printf("CUDA error: %s\n", errorMessage);
                break;
            }
            cudaDeviceSynchronize();
        }
     }
    
    cudaMemcpy(SC,d_min_SC,n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("Minimum Stress Centrality:\n");
    for(int i=0;i<n;i++) printf("%d ",SC[i]);
 
    int oe[n];
    cudaMemcpy(oe,opt_edge,n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("Optimal Edges for respective nodes:\n");
    for(int node=0;node<n;node++){
        int a=oe[node]%n,b=oe[node]/n;
        printf("Node: %d ==> (%d,%d)\n",node,a,b);
    }

    return 0;
}
