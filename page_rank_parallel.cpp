#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <thread>

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
typedef float PageRankType;
#endif

struct thread_data{
  int thread_id;
  long num_vertices;
  long num_edges;
  double barrier1_time;
  double barrier2_time;
  double getNextVertex_time;
  double time_taken;
};

void pageRankSerial(Graph &g, uint max_iters) {
  uintV n = g.n_;

  PageRankType *pr_curr = new PageRankType[n];
  PageRankType *pr_next = new PageRankType[n];

  for (uintV i = 0; i < n; i++) {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  // Push based pagerank
  timer t1;
  double time_taken = 0.0;
  // Create threads and distribute the work across T threads
  // -------------------------------------------------------------------
  t1.start();
  for (uint iter = 0; iter < max_iters; iter++) {
    // for each vertex 'u', process all its outNeighbors 'v'
    for (uintV u = 0; u < n; u++) {
      uintE out_degree = g.vertices_[u].getOutDegree();
      for (uintE i = 0; i < out_degree; i++) {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        pr_next[v] += (pr_curr[u] / out_degree);
      }
    }
    for (uintV v = 0; v < n; v++) {
      pr_next[v] = PAGE_RANK(pr_next[v]);

      // reset pr_curr for the next iteration
      pr_curr[v] = pr_next[v];
      pr_next[v] = 0.0;
    }
  }
  time_taken = t1.stop();
  // -------------------------------------------------------------------

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

void threadFunctionS12(Graph* g, uintV begin, uintV end, PageRankType* pr_curr, std::atomic<PageRankType>* pr_next, uint max_iters, CustomBarrier* my_barrier, thread_data* thread_data){
  timer thread_timer;
  timer barrier1_timer;
  timer barrier2_timer;
  thread_timer.start();

  long local_vertex_count = 0;
  long local_edge_count = 0;
  double barrier1_time = 0.0;
  double barrier2_time = 0.0;

  for (uint iter = 0; iter < max_iters; iter++) {
    for (uintV u = begin; u < end; u++) {
      uintE out_degree = g->vertices_[u].getOutDegree();
      local_edge_count += out_degree;
      for (uintE i = 0; i < out_degree; i++) {
        uintV v = g->vertices_[u].getOutNeighbor(i);
        PageRankType expected = pr_next[v].load();
        PageRankType desired = expected + pr_curr[u] / out_degree;
        while(!pr_next[v].compare_exchange_strong(expected, desired)){
          desired = expected + pr_curr[u] / out_degree;
        }
      }
    }

    barrier1_timer.start();
    my_barrier->wait();
    barrier1_time += barrier1_timer.stop();
    
    for (uintV v = begin; v < end; v++) {
      local_vertex_count++;
      pr_next[v] = PAGE_RANK(pr_next[v].load());

      // reset pr_curr for the next iteration
      pr_curr[v] = pr_next[v];
      pr_next[v] = 0.0;
    }
    
    barrier2_timer.start();
    my_barrier->wait();
    barrier2_time += barrier2_timer.stop();
  }

  double time_taken = thread_timer.stop();
  thread_data->num_vertices = local_vertex_count;
  thread_data->num_edges = local_edge_count;
  thread_data->barrier1_time = barrier1_time;
  thread_data->barrier2_time = barrier2_time;
  thread_data->getNextVertex_time = 0.0;
  thread_data->time_taken = time_taken;
}

void pageRankParallelS1(Graph* g, uint max_iters, uint n_workers) {
  uintV n = g->n_;

  PageRankType *pr_curr = new PageRankType[n];
  std::atomic<PageRankType> *pr_next = new std::atomic<PageRankType>[n];

  for (uintV i = 0; i < n; i++) {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  timer execution_timer;
  timer partition_timer;
  double execution_time = 0.0;
  double partition_time = 0.0;
  execution_timer.start();
  
  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];
  CustomBarrier my_barrier(n_workers);

  uintV begin = 0;
  uintV end = n / n_workers;
  for(uint i = 0; i < n_workers; i++){
    partition_timer.start();
    if(i == n_workers - 1){
      end = n;
    } else {
      end = begin + n / n_workers;
    }
    partition_time += partition_timer.stop();

    threads_data_array[i].thread_id = i;
    threads[i] = std::thread(threadFunctionS12, g, begin, end, pr_curr, pr_next, max_iters, &my_barrier, &threads_data_array[i]);

    begin = end;
  }
  
  for(uint i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();
  std::cout << "thread_id, num_vertices, num_edges, barrier1_time, barrier2_time, getNextVertex_time, total_time\n";
  for(uint i = 0; i < n_workers; i++){
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].barrier1_time << ", " 
              << threads_data_array[i].barrier2_time << ", " 
              << threads_data_array[i].getNextVertex_time << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Partitioning time (in seconds) : " << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << execution_time << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

void pageRankParallelS2(Graph* g, uint max_iters, uint n_workers) {
  uintV n = g->n_;
  uintV m = g->m_;

  PageRankType *pr_curr = new PageRankType[n];
  std::atomic<PageRankType> *pr_next = new std::atomic<PageRankType>[n];

  for (uintV i = 0; i < n; i++) {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  timer execution_timer;
  timer partition_timer;
  double execution_time = 0.0;
  double partition_time = 0.0;
  execution_timer.start();
  
  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];
  CustomBarrier my_barrier(n_workers);

  uintV start_vertex = 0;
  uintV end_vertex = 0;
  uintE edges_per_worker = m / n_workers;
  uintE edges_count = m;

  for(uint i = 0; i < n_workers; i++){
    partition_timer.start();
    uintE worker_edges_count = 0;
    if(i == n_workers - 1) {
      end_vertex = n; 
      worker_edges_count = edges_count;
    } else {
      for(uintV v = start_vertex; v < n; v++) {
        uintE out_degree = g->vertices_[v].getOutDegree();
        worker_edges_count += out_degree;
        if(worker_edges_count >= edges_per_worker) {
          end_vertex = v; 
          
          edges_count -= worker_edges_count;
          edges_per_worker = edges_count / (n_workers - i - 1);
          break;
        }
      }
    }
    partition_time += partition_timer.stop();

    threads_data_array[i].thread_id = i;
    threads[i] = std::thread(threadFunctionS12, g, start_vertex, end_vertex, pr_curr, pr_next, max_iters, &my_barrier, &threads_data_array[i]);

    start_vertex = end_vertex;
  }

  for(uint i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();
  std::cout << "thread_id, num_vertices, num_edges, barrier1_time, barrier2_time, getNextVertex_time, total_time\n";
  for(uint i = 0; i < n_workers; i++){
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].barrier1_time << ", " 
              << threads_data_array[i].barrier2_time << ", " 
              << threads_data_array[i].getNextVertex_time << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Partitioning time (in seconds) : " << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << execution_time << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

std::atomic<uintV> n_count = {0};
uint k;

uintV getNextVertexToBeProcessed(){
  uintV v = n_count.fetch_add(k);
  return v;
}

void threadFunctionS3(Graph* g, PageRankType* pr_curr, std::atomic<PageRankType>* pr_next, uint max_iters, CustomBarrier* my_barrier, thread_data* thread_data){
  timer thread_timer;
  timer barrier1_timer;
  timer barrier2_timer;
  timer getNextVertex_timer;
  thread_timer.start();

  long local_vertex_count = 0;
  long local_edge_count = 0;
  double barrier1_time = 0.0;
  double barrier2_time = 0.0;
  double getNextVertex_time = 0.0;
  uintV n_ = g->n_;

  for (uint iter = 0; iter < max_iters; iter++) {
    while(true) {
      getNextVertex_timer.start();
      uintV u = getNextVertexToBeProcessed();
      getNextVertex_time += getNextVertex_timer.stop();
      if(u >= n_){
        break;
      }
      for (uint j = 0; j < k; j++) {
        uintE out_degree = g->vertices_[u].getOutDegree();
        local_edge_count += out_degree;
        for (uintE i = 0; i < out_degree; i++) {
          uintV v = g->vertices_[u].getOutNeighbor(i);
          PageRankType expected = pr_next[v].load();
          PageRankType desired = expected + pr_curr[u] / out_degree;
          while(!pr_next[v].compare_exchange_strong(expected, desired)){
            desired = expected + pr_curr[u] / out_degree;
          }
        }
        u++;
        if(u >= n_) break;
      }
    }

    barrier1_timer.start();
    my_barrier->wait();
    n_count = 0;
    my_barrier->wait();
    barrier1_time += barrier1_timer.stop();
    
    while(true) {
      getNextVertex_timer.start();
      uintV v = getNextVertexToBeProcessed();
      getNextVertex_time += getNextVertex_timer.stop();
      if(v >= n_){
        break;
      }
      for (uint j = 0; j < k; j++) {
        local_vertex_count++;
        pr_next[v] = PAGE_RANK(pr_next[v].load());

        // reset pr_curr for the next iteration
        pr_curr[v] = pr_next[v];
        pr_next[v] = 0.0;
        v++;
        if(v >= n_) break;
      }
    }

    barrier2_timer.start();
    my_barrier->wait();
    n_count = 0;
    my_barrier->wait();
    barrier2_time += barrier2_timer.stop();
  }

  double time_taken = thread_timer.stop();
  thread_data->num_vertices = local_vertex_count;
  thread_data->num_edges = local_edge_count;
  thread_data->barrier1_time = barrier1_time;
  thread_data->barrier2_time = barrier2_time;
  thread_data->getNextVertex_time = getNextVertex_time;
  thread_data->time_taken = time_taken;
}

void pageRankParallelS3(Graph* g, uint max_iters, uint n_workers, uint granularity) {
  k = granularity;
  uintV n = g->n_;
  
  PageRankType *pr_curr = new PageRankType[n];
  std::atomic<PageRankType> *pr_next = new std::atomic<PageRankType>[n];

  for (uintV i = 0; i < n; i++) {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  timer execution_timer;
  double execution_time = 0.0;
  double partition_time = 0.0;
  execution_timer.start();
  
  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];
  CustomBarrier my_barrier(n_workers);

  for(uint i = 0; i < n_workers; i++){
    threads_data_array[i].thread_id = i;
    threads[i] = std::thread(threadFunctionS3, g, pr_curr, pr_next, max_iters, &my_barrier, &threads_data_array[i]);
  }

  for(uint i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();
  std::cout << "thread_id, num_vertices, num_edges, barrier1_time, barrier2_time, getNextVertex_time, total_time\n";
  for(uint i = 0; i < n_workers; i++){
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].barrier1_time << ", " 
              << threads_data_array[i].barrier2_time << ", " 
              << threads_data_array[i].getNextVertex_time << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++) {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Partitioning time (in seconds) : " << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << execution_time << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "page_rank_push",
      "Calculate page_rank using serial and parallel execution");
  options.add_options(
      "",
      {
          {"nWorkers", "Number of workers",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_WORKERS)},
          {"nIterations", "Maximum number of iterations",
           cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
          {"strategy", "Strategy to be used",
           cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
          {"granularity", "Granularity to be used",
           cxxopts::value<uint>()->default_value(DEFAULT_GRANULARITY)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
      });

  auto cl_options = options.parse(argc, argv);
  uint n_workers = cl_options["nWorkers"].as<uint>();
  uint strategy = cl_options["strategy"].as<uint>();
  uint max_iterations = cl_options["nIterations"].as<uint>();
  uint granularity = cl_options["granularity"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();

#ifdef USE_INT
  std::cout << "Using INT\n";
#else
  std::cout << "Using FLOAT\n";
#endif
  std::cout << std::fixed;
  std::cout << "Number of workers : " << n_workers << "\n";
  std::cout << "Task decomposition strategy : " << strategy << "\n";
  //std::cout << "Iterations : " << max_iterations << "\n";
  std::cout << "Iterations : " << max_iterations << "\n";
  std::cout << "Granularity : " << granularity << "\n";

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";
  switch (strategy) {
  case 0:
    std::cout << "\nSerial\n";
    pageRankSerial(g, max_iterations);
    break;
  case 1:
    std::cout << "\nVertex-based work partitioning\n";
    pageRankParallelS1(&g, max_iterations, n_workers);
    break;
  case 2:
    std::cout << "\nEdge-based work partitioning\n";
    pageRankParallelS2(&g, max_iterations, n_workers);
    break;
  case 3:
    std::cout << "\nDynamic task mapping\n";
    pageRankParallelS3(&g, max_iterations, n_workers, granularity);
    break;
  default:
    break;
  }

  return 0;
}
