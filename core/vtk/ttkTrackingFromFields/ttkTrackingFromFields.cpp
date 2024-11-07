#include <vtkDoubleArray.h>
#include <vtkInformation.h>
#include <vtkPointData.h>

#include <ttkMacros.h>
#include <ttkTrackingFromFields.h>
#include <ttkTrackingFromPersistenceDiagrams.h>
#include <ttkUtils.h>
#include<Timer.h>




vtkStandardNewMacro(ttkTrackingFromFields);

ttkTrackingFromFields::ttkTrackingFromFields() {
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

int ttkTrackingFromFields::FillOutputPortInformation(int port,
                                                     vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid");
    return 1;
  }
  return 0;
}
int ttkTrackingFromFields::FillInputPortInformation(int port,
                                                    vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
    return 1;
  }
  return 0;
}

// (*) Persistence-driven approach
template <class dataType, class triangulationType>
int ttkTrackingFromFields::trackWithPersistenceMatching(
  vtkUnstructuredGrid *output,
  unsigned long fieldNumber,
  const triangulationType *triangulation) {

  ttk::Timer timer;
  double clock = timer.getElapsedTime();
  using trackingTuple = ttk::trackingTuple;

  // 1. get persistence diagrams.
  std::vector<ttk::DiagramType> persistenceDiagrams(fieldNumber);

  this->performDiagramComputation<dataType, triangulationType>(
    (int)fieldNumber, persistenceDiagrams, triangulation);

  double Diagram_RT = timer.getElapsedTime() - clock;
  clock = timer.getElapsedTime();

  // 2. call feature tracking with threshold.
  std::vector<std::vector<ttk::MatchingType>> outputMatchings(fieldNumber - 1);

  double const spacing = Spacing;
  std::string const algorithm = DistanceAlgorithm;
  double const tolerance = Tolerance;
  std::string const wasserstein = WassersteinMetric;

  ttk::TrackingFromPersistenceDiagrams tfp{};
  tfp.setThreadNumber(this->threadNumber_);
  tfp.setDebugLevel(this->debugLevel_);
  tfp.performMatchings(
    (int)fieldNumber, persistenceDiagrams, outputMatchings,
    algorithm, // Not from paraview, from enclosing tracking plugin
    wasserstein, tolerance, PX, PY, PZ, PS, PE // Coefficients
  );

  double Matching_RT = timer.getElapsedTime() - clock;
  clock = timer.getElapsedTime();

  vtkNew<vtkPoints> const points{};
  vtkNew<vtkUnstructuredGrid> const persistenceDiagram{};

  vtkNew<vtkDoubleArray> persistenceScalars{};
  vtkNew<vtkDoubleArray> valueScalars{};
  vtkNew<vtkIntArray> matchingIdScalars{};
  vtkNew<vtkIntArray> lengthScalars{};
  vtkNew<vtkIntArray> timeScalars{};
  vtkNew<vtkIntArray> componentIds{};
  vtkNew<vtkIntArray> pointTypeScalars{};

  persistenceScalars->SetName("Cost");
  valueScalars->SetName("Scalar");
  matchingIdScalars->SetName("MatchingIdentifier");
  lengthScalars->SetName("ComponentLength");
  timeScalars->SetName("TimeStep");
  componentIds->SetName("ConnectedComponentId");
  pointTypeScalars->SetName("CriticalType");

  // (+ vertex id)
  std::vector<trackingTuple> trackingsBase;
  tfp.performTracking(persistenceDiagrams, outputMatchings, trackingsBase);

  double Tracking_RT = timer.getElapsedTime() - clock;
  clock = timer.getElapsedTime();

  std::vector<std::set<int>> trackingTupleToMerged(trackingsBase.size());

  if(DoPostProc) {
    tfp.performPostProcess(persistenceDiagrams, trackingsBase,
                           trackingTupleToMerged, PostProcThresh);
  }

  bool const useGeometricSpacing = UseGeometricSpacing;

  // Build mesh.
  ttkTrackingFromPersistenceDiagrams::buildMesh(
    trackingsBase, outputMatchings, persistenceDiagrams, useGeometricSpacing,
    spacing, DoPostProc, trackingTupleToMerged, points, persistenceDiagram,
    persistenceScalars, valueScalars, matchingIdScalars, lengthScalars,
    timeScalars, componentIds, pointTypeScalars, *this);
    
  output->ShallowCopy(persistenceDiagram);

  double Mesh_RT = timer.getElapsedTime() - clock;
  std::cout<<std::fixed
            <<"DiagramComputationRT = "<<Diagram_RT
            <<", PerformMatchingRT = "<<Matching_RT
            <<", PerformTrackingRT = "<<Tracking_RT
            <<", BuildMeshRT = "<<Mesh_RT<<std::endl; 
  return 1;
}

template<class triangulationType>
  void buildMeshFromTracking(
  const triangulationType *triangulation,
  const std::vector<ttk::trackingTuple> &trackings,
  const bool useGeometricSpacing,
  const double spacing,
  vtkPoints *points,
  vtkUnstructuredGrid *criticalPointTracking,
  vtkIntArray *pointsCriticalType,
  vtkIntArray * timeScalars,
  vtkIntArray *lengthScalars,
  vtkIntArray *globalVertexIds,
  vtkIntArray *connectedComponentIds,
  unsigned int *sizes){

    int pointCpt = 0;
    int edgeCpt=0;
    for (unsigned int i = 0 ; i <  trackings.size() ; i++){
      ttk::CriticalType currentType=ttk::CriticalType::Local_minimum;
      if(i < sizes[0])currentType=ttk::CriticalType::Local_maximum;
      else if(i < sizes[1] && i >=sizes[0])currentType = ttk::CriticalType::Saddle1;
      else if(i < sizes[2] && i >=sizes[1])currentType = ttk::CriticalType::Saddle2; 
      int startTime = std::get<0>(trackings[i]);
      std::vector<ttk::SimplexId> chain = std::get<2>(trackings[i]);

      float x=0;
      float y=0;
      float z=0;
      triangulation->getVertexPoint(chain[0], x, y, z);
      if(useGeometricSpacing)z+=startTime *spacing;
      points->InsertNextPoint(x,y,z);
      globalVertexIds->InsertTuple1(pointCpt, (int)chain[0]);
      pointsCriticalType->InsertTuple1(pointCpt, (int)currentType);
      timeScalars->InsertTuple1(pointCpt, startTime);
      vtkIdType edge[2];
      for (unsigned int j = 1 ; j < chain.size() ; j++){
        triangulation->getVertexPoint(chain[j], x, y, z);
        if(useGeometricSpacing)z+=(j+startTime)*spacing;
        edge[0]=pointCpt;
        pointCpt++;
        edge[1]=pointCpt;
        points->InsertNextPoint(x, y, z);
        globalVertexIds->InsertTuple1(pointCpt, (int)chain[j]);
        criticalPointTracking->InsertNextCell(VTK_LINE, 2, edge);
        pointsCriticalType->InsertTuple1(pointCpt, (int)currentType);
        timeScalars->InsertTuple1(pointCpt, startTime+j);
        lengthScalars->InsertTuple1(edgeCpt, chain.size()-1);
        connectedComponentIds->InsertTuple1(edgeCpt, i);
        edgeCpt++;
      }
      pointCpt++;
    }

    criticalPointTracking->SetPoints(points);
    criticalPointTracking->GetCellData()->AddArray(lengthScalars);
    criticalPointTracking->GetCellData()->AddArray(connectedComponentIds);
    criticalPointTracking->GetPointData()->AddArray(pointsCriticalType);
    criticalPointTracking->GetPointData()->AddArray(timeScalars);
    criticalPointTracking->GetPointData()->AddArray(globalVertexIds);
  }


template <class dataType, class triangulationType>
  int ttkTrackingFromFields::trackWithCriticalPointMatching(vtkUnstructuredGrid *output,
                                   unsigned long fieldNumber,
                                   const triangulationType *triangulation){

    ttk::Timer timer;
    double clock = timer.getElapsedTime();
    ttk::CriticalPointTracking tracker;
    float x, y, z;
    float maxX, minX, maxY, minY, maxZ, minZ;
    triangulation->getVertexPoint(0, minX, minY, minZ );
    triangulation->getVertexPoint(0, maxX, maxY, maxZ );

    for (int i = 0 ; i < triangulation->getNumberOfVertices(); i++){
      triangulation->getVertexPoint(i, x, y, z);
      maxX = std::max(x, maxX);
      maxX = std::min(x, minX);
      maxY = std::max(y, maxX);
      minY = std::min(y, minY);
      maxZ = std::max(z, maxZ);
      minZ = std::min(z, minZ);
    }
    
    double const costDeathBirth = CostDeathBirth;
    double const tolerance = (double)Tolerance;
    float meshDiameter = std::sqrt(std::pow(maxX-minX, 2)  + std::pow(maxY - minY, 2) + std::pow(maxZ - minZ, 2));
    int assignmentMethod = AssignmentMethod;
    tracker.setMeshDiamater(meshDiameter);
    tracker.setTolerance(tolerance);
    tracker.setEpsilon(costDeathBirth);
    tracker.setAssignmentMethod(assignmentMethod);
    tracker.setWeights(PX, PY, PZ, PF);
    tracker.setThreadNumber(this->threadNumber_);
    
    std::vector<ttk::DiagramType> persistenceDiagrams(fieldNumber);
    this->setDebugLevel(10);
    this->performDiagramComputation<dataType, triangulationType>((int)fieldNumber, persistenceDiagrams, triangulation);

    double Diagram_RT = timer.getElapsedTime() - clock;
    clock = timer.getElapsedTime();
    std::vector<std::vector<ttk::MatchingType>> maximaMatchings(fieldNumber-1);
    std::vector<std::vector<ttk::MatchingType>> sad_1_Matchings(fieldNumber-1);
    std::vector<std::vector<ttk::MatchingType>> sad_2_Matchings(fieldNumber-1);
    std::vector<std::vector<ttk::MatchingType>> minimaMatchings(fieldNumber-1);    

    tracker.performMatchings(persistenceDiagrams, 
                              maximaMatchings,
                              sad_1_Matchings,
                              sad_2_Matchings,
                              minimaMatchings, 
                              fieldNumber);
  
    double Matching_RT = timer.getElapsedTime() - clock;
    clock = timer.getElapsedTime();
    vtkNew<vtkPoints> const points{};
    vtkNew<vtkUnstructuredGrid> const criticalPointTracking{};

    vtkNew<vtkDoubleArray> costs{};
    vtkNew<vtkDoubleArray> valueScalars{};
    vtkNew<vtkIntArray> globalVertexIds{};
    vtkNew<vtkIntArray> lengthScalars{};
    vtkNew<vtkIntArray> timeScalars{};
    vtkNew<vtkIntArray> connectedComponentIds{};
    vtkNew<vtkIntArray> pointsCriticalType{};

    costs->SetName("Cost");
    valueScalars->SetName("Scalar");
    globalVertexIds->SetName("VertexGlobalId");
    lengthScalars->SetName("ComponentLength");
    timeScalars->SetName("TimeStep");
    connectedComponentIds->SetName("ConnectedComponentId");
    pointsCriticalType->SetName("CriticalType");

    std::vector<ttk::trackingTuple> allTrackings;
    unsigned int typesArrayLimits [3]={};
    tracker.performTrackings(
        fieldNumber,
        maximaMatchings,
        sad_1_Matchings,
        sad_2_Matchings,
        minimaMatchings,
        allTrackings,
        typesArrayLimits);

    double Tracking_RT = timer.getElapsedTime() - clock;
    clock = timer.getElapsedTime();

    double const spacing = Spacing;
    bool const useGeometricSpacing = UseGeometricSpacing;

    buildMeshFromTracking(
      triangulation,
      allTrackings,
      useGeometricSpacing, spacing, 
      points, 
      criticalPointTracking,
      pointsCriticalType,
      timeScalars,
      lengthScalars,
      globalVertexIds,
      connectedComponentIds,
      typesArrayLimits);

    output->ShallowCopy(criticalPointTracking);

    double Mesh_RT = timer.getElapsedTime() - clock;
    std::cout<<std::fixed
          <<"DiagramComputationRT = "<<Diagram_RT
          <<", PerformMatchingRT = "<<Matching_RT
          <<", PerformTrackingRT = "<<Tracking_RT
          <<", BuildMeshRT = "<<Mesh_RT<<std::endl;
    return 1;
}

int ttkTrackingFromFields::RequestData(vtkInformation *ttkNotUsed(request),
                                       vtkInformationVector **inputVector,
                                       vtkInformationVector *outputVector) {


  ttk::Timer timer;

  auto input = vtkDataSet::GetData(inputVector[0]);
  auto output = vtkUnstructuredGrid::GetData(outputVector);
  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(input);
  if(!triangulation)
    return 0;

  this->preconditionTriangulation(triangulation);


  // Test validity of datasets
  if(input == nullptr || output == nullptr) {
    return -1;
  }

  // Get number and list of inputs.
  std::vector<vtkDataArray *> inputScalarFieldsRaw;
  std::vector<vtkDataArray *> inputScalarFields;
  const auto pointData = input->GetPointData();
  int numberOfInputFields = pointData->GetNumberOfArrays();
  if(numberOfInputFields < 3) {
    this->printErr("Not enough input fields to perform tracking.");
  }

  vtkDataArray *firstScalarField = pointData->GetArray(0);

  for(int i = 0; i < numberOfInputFields; ++i) {
    vtkDataArray *currentScalarField = pointData->GetArray(i);
    if(currentScalarField == nullptr
       || currentScalarField->GetName() == nullptr) {
      continue;
    }
    std::string const sfname{currentScalarField->GetName()};
    if(sfname.rfind("_Order") == (sfname.size() - 6)) {
      continue;
    }
    if(firstScalarField->GetDataType() != currentScalarField->GetDataType()) {
      this->printErr("Inconsistent field data type or size between fields `"
                     + std::string{firstScalarField->GetName()} + "' and `"
                     + sfname + "'");
      return -1;
    }
    inputScalarFieldsRaw.push_back(currentScalarField);
  }

  std::sort(inputScalarFieldsRaw.begin(), inputScalarFieldsRaw.end(),
            [](vtkDataArray *a, vtkDataArray *b) {
              std::string s1 = a->GetName();
              std::string s2 = b->GetName();
              return std::lexicographical_compare(
                s1.begin(), s1.end(), s2.begin(), s2.end());
            });

  numberOfInputFields = inputScalarFieldsRaw.size();
  int const end = EndTimestep <= 0 ? numberOfInputFields
                                   : std::min(numberOfInputFields, EndTimestep);
  for(int i = StartTimestep; i < end; i += Sampling) {
    vtkDataArray *currentScalarField = inputScalarFieldsRaw[i];
    // Print scalar field names:
    // std::cout << currentScalarField->GetName() << std::endl;
    inputScalarFields.push_back(currentScalarField);
  }

  // Input -> persistence filter.
  std::string const algorithm = DistanceAlgorithm;
  int const pvalg = PVAlgorithm;
  bool useTTKMethod = false;
  bool criticalPointTracking = (pvalg == 2);


  if(pvalg >= 0) {
    switch(pvalg) {
      case 0:
      case 1:
      case 2:
      case 3:
        useTTKMethod = true;
        break;
      case 4:
        break;
      default:
        this->printMsg("Unrecognized tracking method.");
        break;
    }
  } else {
    using ttk::str2int;
    switch(str2int(algorithm.c_str())) {
      case str2int("0"):
      case str2int("ttk"):
      case str2int("1"):
      case str2int("legacy"):
      case str2int("2"):
      case str2int("geometric"):
      case str2int("3"):
      case str2int("parallel"):
        useTTKMethod = true;
        break;
      case str2int("4"):
      case str2int("greedy"):
        break;
      default:
        this->printMsg("Unrecognized tracking method.");
        break;
    }
  }

  // 0. get data
  int const fieldNumber = inputScalarFields.size();
  std::vector<void *> inputFields(fieldNumber);
  for(int i = 0; i < fieldNumber; i++) {
    inputFields[i] = ttkUtils::GetVoidPointer(inputScalarFields[i]);
  }
  this->setInputScalars(inputFields);

  // 0'. get offsets
  std::vector<ttk::SimplexId *> inputOrders(fieldNumber);
  for(int i = 0; i < fieldNumber; ++i) {
    this->SetInputArrayToProcess(0, 0, 0, 0, inputScalarFields[i]->GetName());
    auto orderArray
      = this->GetOrderArray(input, 0, triangulation, false, 0, false);
    inputOrders[i]
      = static_cast<ttk::SimplexId *>(ttkUtils::GetVoidPointer(orderArray));
  }
  this->setInputOffsets(inputOrders);

  double clock = timer.getElapsedTime(); 

  int status = 0;
  if(useTTKMethod && !criticalPointTracking) {
    ttkVtkTemplateMacro(
      inputScalarFields[0]->GetDataType(), triangulation->getType(),
      (status = this->trackWithPersistenceMatching<VTK_TT, TTK_TT>(
         output, fieldNumber, (TTK_TT *)triangulation->getData())));
  } 
  else if(useTTKMethod && criticalPointTracking){
    ttkVtkTemplateMacro(
      inputScalarFields[0]->GetDataType(), triangulation->getType(),
      (status = this->trackWithCriticalPointMatching<VTK_TT, TTK_TT>(
         output, fieldNumber, (TTK_TT *)triangulation->getData())));
  } else {
    this->printMsg("The specified matching method is not supported.");
  }

  double total_rt = timer.getElapsedTime() - clock;
  std::cout<<"TrackingMethodTime = "<<total_rt<<std::endl;
  return status;
}
