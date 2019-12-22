#include <igl/signed_distance.h>
#include <igl/read_triangle_mesh.h>
#include <igl/fast_winding_number.h>
#include <cmath>
#include <ctime>
#include "miniball.hpp"

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>



// global mesh data
Eigen::MatrixXd V;
Eigen::MatrixXi F;
igl::AABB<Eigen::MatrixXd,3> tree;
igl::FastWindingNumberBVH fwn_bvh;

void sphere_normalization(Eigen::MatrixXd &V, float target_radius) {
  typedef double mytype;
  int n = V.rows();
  int d = 3;

  mytype** ap = new mytype*[n];
  // lets fill ap with our points
  for (int i=0; i<n; ++i) {
    mytype* p = new mytype[d];
    for (int j=0; j<d; ++j) {
      p[j] = V.coeff(i,j);  
    }
    ap[i]=p;
  }

  typedef mytype* const* PointIterator; 
  typedef const mytype* CoordIterator;
  typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;
  MB mb (3, ap, ap+n);

  // radius can be used to scale the shape!
  float radius = sqrt(mb.squared_radius()); 
  float scale = target_radius/radius;

  Eigen::Affine3d T = Eigen::Affine3d::Identity();
  T.translation() <<
    -mb.center()[0]*scale, 
    -mb.center()[1]*scale, 
    -mb.center()[2]*scale;

  T.scale(scale);

  for( auto i=0; i < V.rows(); ++i){
    V.row(i).transpose() = T.linear()*V.row(i).transpose() + T.translation();
  }

  // cleanup
  for(int j = 0; j < n; j++) {
    delete ap[j];
  }
  delete [] ap;
}

void triangleMeshLoader(const char * inputFilePath, float r) {
  igl::read_triangle_mesh(inputFilePath, V, F);
  sphere_normalization(V, r);
  // Precompute signed distance AABB tree
  tree.init(V,F);
  // precompute fwn bvh... 
  igl::fast_winding_number(V.template cast<float>(), F, 2, fwn_bvh);
}

Eigen::VectorXd querySDF(Eigen::MatrixXf &V_queries) {
  Eigen::VectorXd S;
  Eigen::MatrixXd queries = V_queries.cast<double> ();
  signed_distance_fast_winding_number(queries, V, F, tree, fwn_bvh, S);
  return S;
}

// N: number of points, r: sphere radius
Eigen::MatrixXf generateQueryPoints(int N, float r) {
  Eigen::MatrixXf queryPts(N,3);
  int validPts = 0;

  while (validPts < N) {
    int remaining = N - validPts;

    Eigen::MatrixXf pts = Eigen::MatrixXf::Random(remaining, 3); //random -1 to 1
    //rejection criteria
    for (int i =0; i < pts.rows(); i++) {
      if (pts.row(i).norm() <= r) {
        queryPts.row(validPts) = pts.row(i);
        validPts ++;
      }      
    }
  }

  return queryPts;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
      return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
  return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[])
{
  float r = 1.0; //default to unit sphere
  int N = pow(10,7); //default to 100 million samples
  const char * inputFilePath = "";
  const char * outputFilePath = "";
  const char * sampleFilePath = "";
  
  if(cmdOptionExists(argv, argv+argc, "-i"))
  {
    inputFilePath = getCmdOption(argv, argv + argc, "-i");
  } 

  if(cmdOptionExists(argv, argv+argc, "-o"))
  {
    outputFilePath = getCmdOption(argv, argv + argc, "-o");
  } 

  if(cmdOptionExists(argv, argv+argc, "-r"))
  {
    r = atof(getCmdOption(argv, argv + argc, "-r"));
  }

  if(cmdOptionExists(argv, argv+argc, "-N"))
  {
    N = atoi(getCmdOption(argv, argv + argc, "-N"));
  }

  // use h5 dataset of queries! (instead of generating random!)
  if (cmdOptionExists(argv, argv+argc, "-s")) {
    sampleFilePath =  getCmdOption(argv, argv + argc, "-s");
  }

  Eigen::MatrixXf V_queries;

  // Query Record Creation Mode
  if (!cmdOptionExists(argv, argv+argc, "-i") && !cmdOptionExists(argv, argv+argc, "-o") && cmdOptionExists(argv, argv+argc, "-s") ) {
    HighFive::File queryFile(sampleFilePath, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    V_queries = generateQueryPoints(N, r);

    std::vector<size_t> dims (2);
    dims[0] = V_queries.rows();
    dims[1] = V_queries.cols();

    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking({dims[0],dims[1]}));
    props.add(HighFive::Deflate(9));  // enable compression

    std::vector<std::vector<float> > pts(dims[0], std::vector<float>(dims[1]));

    for (int r = 0; r < dims[0]; r ++) {
      std::vector<float> p(V_queries.row(r).data(), V_queries.row(r).data() + V_queries.row(r).cols());
      pts[r] = p;
    }

    HighFive::DataSet Q = queryFile.createDataSet<float>("queries", HighFive::DataSpace::From(pts), props);
    Q.write(pts);
    return 1;
  }


  if (cmdOptionExists(argv, argv+argc, "-i") && cmdOptionExists(argv, argv+argc, "-o")) {
    if (strcmp(sampleFilePath,"") == 0) {
      V_queries = generateQueryPoints(N, r);
    } else {
      // read queries from h5
      HighFive::File queryFile(sampleFilePath, HighFive::File::ReadWrite);
      HighFive::DataSet queryDataset = queryFile.getDataSet("queries");
      std::vector<std::vector<float>> queries;
      std::vector<size_t> dim = queryDataset.getDimensions();

      V_queries.resize(dim[0], dim[1]);

      queryDataset.read(queries);

      for (int i = 0; i < queries.size(); i ++) {
        V_queries.row(i) = Eigen::Map<Eigen::VectorXf>(queries[i].data(), queries[i].size());
      }
    }

    // now load the mesh
    triangleMeshLoader(inputFilePath, r * 0.9); //normalize to be "just inside"
    // now query he SDF values for mesh, given queries
    Eigen::VectorXd S = querySDF(V_queries);

    // save to file
    HighFive::File sdfFile(outputFilePath, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
    HighFive::DataSpace dataspace(S.size());
    HighFive::DataSetCreateProps props;

    props.add(HighFive::Chunking(S.size()));
    props.add(HighFive::Deflate(9));  // enable compression
    
    HighFive::DataSet datasetS = sdfFile.createDataSet<float>("sdf", dataspace, props);
    datasetS.write(S.data());

  } else {
    std::cout << "I dont think you're using this right...\n";
    std::cout << "For generating sdf from a given h5 of queries \n";
    std::cout << "./sdfDataGen -i MESH_PATH -o H5_PATH -s QUERY_PATH \n";
    std::cout << "For generating h5 of queries\n";
    std::cout << "./sdfDataGen -s queries.h5 -r 1.0 -N 100\n";
    return 0;
  }
  
  return 1;

}
