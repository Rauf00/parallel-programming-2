#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <atomic>

struct thread_data{
  int thread_id;
  long num_vertices;
  long num_edges;
  long triangle_count;
  double time_taken;
};

long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2,
                    uintV u, uintV v) {

  uintE i = 0, j = 0; // indexes for array1 and array2
  long count = 0;

  if (u == v)
    return count;

  while ((i < len1) && (j < len2)) {
    if (array1[i] == array2[j]) {
      if ((array1[i] != u) && (array1[i] != v)) {
        count++;
      } else {
        // triangle with self-referential edge -> ignore
      }
      i++;
      j++;
    } else if (array1[i] < array2[j]) {
      i++;
    } else {
      j++;
    }
  }
  return count;
}

void triangleCountSerial(Graph &g) {
  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  t1.start();
  for (uintV u = 0; u < n; u++) {
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }
  time_taken = t1.stop();
  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << time_taken << "\n";
}

void threadFunctionS1(Graph* g, uintV begin, uintV end, thread_data* thread_data){
  timer thread_timer;
  thread_timer.start();
  long local_edge_count = 0;
  long local_triangle_count = 0;

  for (uintV u = begin; u < end; u++) {
    uintE out_degree = g->vertices_[u].getOutDegree();
    local_edge_count += out_degree;
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g->vertices_[u].getOutNeighbor(i);
      local_triangle_count += countTriangles(g->vertices_[u].getInNeighbors(),
                                       g->vertices_[u].getInDegree(),
                                       g->vertices_[v].getOutNeighbors(),
                                       g->vertices_[v].getOutDegree(), u, v);
    }
  }
  double time_taken = thread_timer.stop();
  thread_data->triangle_count = local_triangle_count;
  thread_data->num_vertices = end - begin;
  thread_data->num_edges = local_edge_count;
  thread_data->time_taken = time_taken;
}

void triangleCountParallelS1(Graph* g, int n_workers) {
  uintV n = g->n_;
  long triangle_count = 0;

  double execution_time = 0.0;
  double partition_time = 0.0;
  timer execution_timer;
  timer partition_timer;
  execution_timer.start();

  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];

  uintV begin = 0;
  uintV end = n / n_workers;
  for(int i = 0; i < n_workers; i++){
    partition_timer.start();
    if(i == n_workers - 1){
      end = n;
    } else {
      end = begin + n / n_workers;
    }
    partition_time += partition_timer.stop();

    threads_data_array[i].thread_id = i;
    threads[i] = std::thread(threadFunctionS1, g, begin, end, &threads_data_array[i]);

    begin = end;
  }

  // Join threads
  for(uint i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();

  std::cout << "thread_id, num_vertices, num_edges, triangle_count, time_taken\n";
  for(int i = 0; i < n_workers; i++){
    triangle_count += threads_data_array[i].triangle_count;
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].triangle_count << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Partitioning time (in seconds) : " << std::setprecision(TIME_PRECISION)
            << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << execution_time << "\n";
}

void threadFunctionS2(Graph* g, uintV begin, uintV end, thread_data* thread_data){
  timer thread_timer;
  thread_timer.start();

  long local_triangle_count = 0;
  for (uintV u = begin; u < end; u++) {
    uintE out_degree = g->vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g->vertices_[u].getOutNeighbor(i);
      local_triangle_count += countTriangles(g->vertices_[u].getInNeighbors(),
                                       g->vertices_[u].getInDegree(),
                                       g->vertices_[v].getOutNeighbors(),
                                       g->vertices_[v].getOutDegree(), u, v);
    }
  }

  double time_taken = thread_timer.stop();

  thread_data->triangle_count = local_triangle_count;
  thread_data->time_taken = time_taken;
}

void triangleCountParallelS2(Graph* g, int n_workers) {
  uintE n = g->n_;
  uintE m = g->m_;
  long triangle_count = 0;

  double execution_time = 0.0;
  double partition_time = 0.0;
  timer execution_timer;
  timer partition_timer;
  execution_timer.start();

  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];

  uintV start_vertex = 0;
  uintV end_vertex = 0;
  uintE edges_per_worker = m / n_workers;
  uintE edges_count = m;

  for(int i = 0; i < n_workers; i++){
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
    threads_data_array[i].num_vertices = 0;
    threads_data_array[i].num_edges = worker_edges_count;
    threads[i] = std::thread(threadFunctionS2, g, start_vertex, end_vertex, &threads_data_array[i]);
    start_vertex = end_vertex;
  }

  // Join threads
  for(uint i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();

  std::cout << "thread_id, num_vertices, num_edges, triangle_count, time_taken\n";
  for(int i = 0; i < n_workers; i++){
    triangle_count += threads_data_array[i].triangle_count;
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].triangle_count << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Partitioning time (in seconds) : " << std::setprecision(TIME_PRECISION)
            << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << execution_time << "\n";
}

std::atomic<uintV> n_count;

uintV getNextVertexToBeProcessed(){
  uintV v = n_count.fetch_add(-1);
  return v;
}

void threadFunctionS3(Graph* g, thread_data* thread_data, double* partition_time){
  timer thread_timer;
  timer partition_timer;
  thread_timer.start();
  long local_vertex_count = 0;
  long local_edge_count = 0;
  long local_triangle_count = 0;
  double partition_time_t0 = 0.0;

  uintV u;
  while(true){
    partition_timer.start();
    uintV u = getNextVertexToBeProcessed();
    double t_partitionTime = partition_timer.stop();
    if(thread_data->thread_id == 0){
      partition_time_t0 += t_partitionTime;
    }

    if(u <= -1){
      break;
    }
    local_vertex_count++;
    uintE out_degree = g->vertices_[u].getOutDegree();
    local_edge_count += out_degree;

    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g->vertices_[u].getOutNeighbor(i);
      local_triangle_count += countTriangles(g->vertices_[u].getInNeighbors(),
                                       g->vertices_[u].getInDegree(),
                                       g->vertices_[v].getOutNeighbors(),
                                       g->vertices_[v].getOutDegree(), u, v);
    }
  }
  double time_taken = thread_timer.stop();
  *partition_time = partition_time_t0;

  thread_data->triangle_count = local_triangle_count;
  thread_data->num_vertices = local_vertex_count;
  thread_data->num_edges = local_edge_count;
  thread_data->time_taken = time_taken;
}

void triangleCountParallelS3(Graph* g, int n_workers) {
  n_count = g->n_ - 1;
  long triangle_count = 0;

  double execution_time = 0.0;
  double partition_time = 0.0;
  timer execution_timer;
  timer partition_timer;
  execution_timer.start();

  std::thread threads[n_workers];
  struct thread_data threads_data_array[n_workers];

  for(int i = 0; i < n_workers; i++){
    threads_data_array[i].thread_id = i;
    threads[i] = std::thread(threadFunctionS3, g, &threads_data_array[i], &partition_time);
  }

  // Join threads
  for(int i = 0; i < n_workers; i++){
    threads[i].join();
  }

  execution_time = execution_timer.stop();

  std::cout << "thread_id, num_vertices, num_edges, triangle_count, time_taken\n";
  for(int i = 0; i < n_workers; i++){
    triangle_count += threads_data_array[i].triangle_count;
    std::cout << threads_data_array[i].thread_id << ", " 
              << threads_data_array[i].num_vertices << ", " 
              << threads_data_array[i].num_edges << ", " 
              << threads_data_array[i].triangle_count << ", " 
              << threads_data_array[i].time_taken << "\n";
  }

  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Partitioning time (in seconds) : " << std::setprecision(TIME_PRECISION)
            << partition_time << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << execution_time << "\n";
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "triangle_counting_serial",
      "Count the number of triangles using serial and parallel execution");
  options.add_options(
      "custom",
      {
          {"nWorkers", "Number of workers",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_WORKERS)},
          {"strategy", "Strategy to be used",
           cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
      });

  auto cl_options = options.parse(argc, argv);
  uint n_workers = cl_options["nWorkers"].as<uint>();
  uint strategy = cl_options["strategy"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  std::cout << std::fixed;
  std::cout << "Number of workers : " << n_workers << "\n";
  std::cout << "Task decomposition strategy : " << strategy << "\n";

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";

  switch (strategy) {
  case 0:
    std::cout << "\nSerial\n";
    triangleCountSerial(g);
    break;
  case 1:
    std::cout << "\nVertex-based work partitioning\n";
    triangleCountParallelS1(&g, n_workers);
    break;
  case 2:
    std::cout << "\nEdge-based work partitioning\n";
    triangleCountParallelS2(&g, n_workers);
    break;
  case 3:
    std::cout << "\nDynamic task mapping\n";
    triangleCountParallelS3(&g, n_workers);
    break;
  default:
    break;
  }

  return 0;
}
