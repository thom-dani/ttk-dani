#include <CriticalPointTracking.h>

//Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are critical points

double ttk::CriticalPointTracking::criticalPointDistance(
    const std::array<float, 3> coords_p1,
    const double sfValue_p1,
    const std::array<float, 3> coords_p2,
    const double sfValue_p2,
    int p = 2){
    return std::pow(std::pow(coords_p1[0] - coords_p2[0], p)
                    + std::pow(coords_p1[1] - coords_p2[1], p)
                    + std::pow(coords_p1[2] - coords_p2[2], p)
                    + std::pow(sfValue_p1 - sfValue_p2,p), 1.0/p);
}


//Sort the critical points by types

void ttk::CriticalPointTracking::sortCriticalPoint(
    const DiagramType &d, #include <CriticalPointTracking.h>

//Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are critical points

double ttk::CriticalPointTracking::criticalPointDistance(
    const std::array<float, 3> coords_p1,
    const double sfValue_p1,
    const std::array<float, 3> coords_p2,
    const double sfValue_p2,
    int p = 2){
    return std::pow(std::pow(coords_p1[0] - coords_p2[0], p)
                    + std::pow(coords_p1[1] - coords_p2[1], p)
                    + std::pow(coords_p1[2] - coords_p2[2], p)
                    + std::pow(sfValue_p1 - sfValue_p2,p), 1.0/p);
}


//Sort the critical points by types

void ttk::CriticalPointTracking::sortCriticalPoint(
    const DiagramType &d, 
    std::vector<SimplexId> maxId,
    std::vector<SimplexId> sad_1_Id,
    std::vector<SimplexId> sad_2_Id,
    std::vector<SimplexId> minId)
    {
        for (unsigned int i = 0 ; i < d.size() ; i++ ){
          SimplexId birthId = d[i].birth.id;
          SimplexId deathId = d[i].death.id;
            switch(d[i].birth.type){
                case CriticalType::Local_maximum:
                    maxId.push_back(birthId);
                case CriticalType::Saddle1:
                    sad_1_Id.push_back(birthId);
                case CriticalType::Saddle2:
                    sad_2_Id.push_back(birthId);
                case CriticalType::Local_minimum:
                    minId.push_back(birthId);
                default : 
                    break;
            }
            switch(d[i].death.type){
                case CriticalType::Local_maximum:
                    maxId.push_back(deathId);
                case CriticalType::Saddle1:
                    sad_1_Id.push_back(deathId);
                case CriticalType::Saddle2:
                    sad_2_Id.push_back(deathId);
                case CriticalType::Local_minimum:
                    minId.push_back(deathId);
                default : 
                    break;
            }
        }
    }  


void ttk::CriticalPointTracking::buildCostMatrix(
  const std::vector<std::array<float, 3>> coords_1,
  const std::vector<double> sfValues_1,
  const std::vector<std::array<float, 3>> coords_2,
  const std::vector<double> sfValues_2,
  std::vector<std::vector<double>> &matrix,
  float costDeathBirth
  )
  {
      int size_1 = coords_1.size();
      int size_2 = coords_2.size();
      int matrix_size = size_1 + size_2;
      for (int i = 0 ; i < size_1 ; i++){
          for (int j = 0 ; j < size_2 ; j++){
              matrix[i][j]=criticalPointDistance(coords_1[i], sfValues_1[i], coords_2[i], sfValues_2[i]);
          }
      }
      for (int i = size_1 ; i < matrix_size ; i++){
          matrix[i][i-size_1] = costDeathBirth;
      }
      for (int i = size_2 ; i < matrix_size ; i++){
          matrix[i-size_2][i] = costDeathBirth;
      }
  }

void ttk::CriticalPointTracking::buildCostMatrices(
  const std::vector<SimplexId> &idDiagram_1,
  const std::vector<SimplexId> &idDiagram_2,
  const DiagramType &d1,
  const DiagramType &d2,
  std::vector<std::vector<double>> &costMatrix)
  {
      std::vector<std::array<float, 3 >> maximaCoords_1; 
      std::vector<std::array<float, 3 >> minimaCoords_1;
      std::vector<std::array<float, 3 >> sad_1_Coords_1; 
      std::vector<std::array<float, 3 >> sad_2_Coords_1;
      std::vector<double> maximaScalarValues_1;
      std::vector<double> sad_1_ScalarValues_1;
      std::vector<double> sad_2_ScalarValues_1;
      std::vector<double> minimaScalarValues_1;

      int nb_row_max = maximaScalarValues_1.size();
      int nb_row_sad_1 = sad_1_ScalarValues_1.size();
      int nb_row_sad_2 = sad_2_ScalarValues_1.size(); 
      int nb_row_min = minimaScalarValues_1.size();

      std::vector<std::array<float, 3 >> maximaCoords_2; 
      std::vector<std::array<float, 3 >> minimaCoords_2;
      std::vector<std::array<float, 3 >> sad_1_Coords_2; 
      std::vector<std::array<float, 3 >> sad_2_Coords_2;
      std::vector<double> maximaScalarValues_2;
      std::vector<double> minimaScalarValues_2;
      std::vector<double> sad_1_ScalarValues_2;
      std::vector<double> sad_2_ScalarValues_2;

      int nb_col_max = maximaScalarValues_2.size();
      int nb_col_sad_1 = sad_1_ScalarValues_2.size();
      int nb_col_sad_2 = sad_2_ScalarValues_2.size();
      int nb_col_min = minimaScalarValues_2.size();

      maxMatrix.resize(nb_row_max);
      for (int i = 0 ; i < nb_row_max ; i++){
        maxMatrix[i].resize(nb_col_max);
      }
      sad_1_Matrix.resize(nb_row_sad_1);
      for (int i = 0 ; i < nb_row_sad_1 ; i++){
        sad_1_Matrix[i].resize(nb_col_sad_1);
      }
      sad_2_Matrix.resize(nb_row_sad_2);
      for (int i = 0 ; i < nb_row_sad_2 ; i++){
        sad_2_Matrix[i].resize(nb_col_sad_2);
      }
      minMatrix.resize(nb_row_min);
      for (int i = 0 ; i < nb_row_min ; i++){
        minMatrix[i].resize(nb_col_min);
      }

      float costDeathBirth = epsilon*std::sqrt(computeScalarBoundingBoxSqrd(d1, d2) + boundingBoxRadiusSqrd);

      buildCostMatrix(maximaCoords_1, maximaScalarValues_1, maximaCoords_2, maximaScalarValues_2, maxMatrix, costDeathBirth);
      buildCostMatrix(sad_1_Coords_1, sad_1_ScalarValues_1, sad_1_Coords_2, sad_1_ScalarValues_2, sad_1_Matrix, costDeathBirth);
      buildCostMatrix(sad_2_Coords_1, sad_2_ScalarValues_1, sad_2_Coords_2, sad_2_ScalarValues_2, sad_2_Matrix, costDeathBirth);
      buildCostMatrix(minimaCoords_1, minimaScalarValues_1, minimaCoords_2, minimaScalarValues_2, minMatrix, costDeathBirth);
  }


  void ttk::CriticalPointTracking::auctionAssignement(
      std::vector<std::vector<double>> &costMatrix,
      std::vector<ttk::MatchingType> &matching)
  {
      ttk::AssignmentAuction<double> solver;
      solver.setInput(costMatrix);
      solver.run(matching);
      solver.clearMatrix();
  }



  void ttk::CriticalPointTracking::performMatchings(
    std::vector<DiagramType> persistenceDiagrams, 
    std::vector<std::vector<MatchingType>> &maximaMatchings,
    std::vector<std::vector<MatchingType>> &sad_1_Matchings,
    std::vector<std::vector<MatchingType>> &sad_2_Matchings,
    std::vector<std::vector<MatchingType>> &minimaMatchings, 
    int fieldNumber)
    {
      for (int i = 0 ; i < fieldNumber-1 ; i++){

        std::vector<std::vector<double>> maxMatrix;
        std::vector<std::vector<double>> sad_1_Matrix;
        std::vector<std::vector<double>> sad_2_Matrix;
        std::vector<std::vector<double>> minMatrix;

        std::vector<SimplexId> maxId_1;
        std::vector<SimplexId> sad_1_Id_1;
        std::vector<SimplexId> sad_2_Id_1;
        std::vector<SimplexId> minId_1;
        sortCriticalPoint(persistenceDiagrams[i], maxId_1, sad_1_Id_1, sad_2_Id_1, minId_1);

        std::vector<SimplexId> maxId_2;
        std::vector<SimplexId> sad_1_Id_2;
        std::vector<SimplexId> sad_2_Id_2;
        std::vector<SimplexId> minId_2;
        sortCriticalPoint(persistenceDiagrams[i+1], maxId_2, sad_1_Id_2, sad_2_Id_2, minId_2);

        buildCostMatrices(maxMatrix, sad_1_Matrix, sad_2_Matrix, minMatrix, persistenceDiagrams[i], persistenceDiagrams[i+1]);

        std::vector<ttk::MatchingType> maxMatching;
        std::vector<ttk::MatchingType> sad_1_Matching;
        std::vector<ttk::MatchingType> sad_2_Matching;
        std::vector<ttk::MatchingType> minMatching;

        auctionAssignement(maxMatrix, maxMatching);
        auctionAssignement(sad_1_Matrix, sad_1_Matching);
        auctionAssignement(sad_2_Matrix, sad_2_Matching);
        auctionAssignement(minMatrix, minMatching);

        maximaMatchings.push_back(maxMatching);
        sad_1_Matchings.push_back(sad_1_Matching);
        sad_1_Matchings.push_back(sad_2_Matching);
        minimaMatchings.push_back(minMatching);

      }
    }

    std::vector<SimplexId> maxId,
    std::vector<SimplexId> sad_1_Id,
    std::vector<SimplexId> sad_2_Id,
    std::vector<SimplexId> minId)
    {
        for (unsigned int i = 0 ; i < d.size() ; i++ ){
          SimplexId birthId = d[i].birth.id;
          SimplexId deathId = d[i].death.id;
            switch(d[i].birth.type){
                case CriticalType::Local_maximum:
                    maxId.push_back(birthId);
                case CriticalType::Saddle1:
                    sad_1_Id.push_back(birthId);
                case CriticalType::Saddle2:
                    sad_2_Id.push_back(birthId);
                case CriticalType::Local_minimum:
                    minId.push_back(birthId);
                default : 
                    break;
            }
            switch(d[i].death.type){
                case CriticalType::Local_maximum:
                    maxId.push_back(deathId);
                case CriticalType::Saddle1:
                    sad_1_Id.push_back(deathId);
                case CriticalType::Saddle2:
                    sad_2_Id.push_back(deathId);
                case CriticalType::Local_minimum:
                    minId.push_back(deathId);
                default : 
                    break;
            }
        }
    }  


void ttk::CriticalPointTracking::buildCostMatrix(
  const std::vector<std::array<float, 3>> coords_1,
  const std::vector<double> sfValues_1,
  const std::vector<std::array<float, 3>> coords_2,
  const std::vector<double> sfValues_2,
  std::vector<std::vector<double>> &matrix,
  float costDeathBirth
  )
  {
      int size_1 = coords_1.size();
      int size_2 = coords_2.size();
      int matrix_size = size_1 + size_2;
      for (int i = 0 ; i < size_1 ; i++){
          for (int j = 0 ; j < size_2 ; j++){
              matrix[i][j]=criticalPointDistance(coords_1[i], sfValues_1[i], coords_2[i], sfValues_2[i]);
          }
      }
      for (int i = size_1 ; i < matrix_size ; i++){
          matrix[i][i-size_1] = costDeathBirth;
      }
      for (int i = size_2 ; i < matrix_size ; i++){
          matrix[i-size_2][i] = costDeathBirth;
      }
  }

void ttk::CriticalPointTracking::buildCostMatrices(
  const std::vector<SimplexId> &idDiagram_1,
  const std::vector<SimplexId> &idDiagram_2,
  const DiagramType &d1,
  const DiagramType &d2,
  std::vector<std::vector<double>> &costMatrix)
  {
      std::vector<std::array<float, 3 >> maximaCoords_1; 
      std::vector<std::array<float, 3 >> minimaCoords_1;
      std::vector<std::array<float, 3 >> sad_1_Coords_1; 
      std::vector<std::array<float, 3 >> sad_2_Coords_1;
      std::vector<double> maximaScalarValues_1;
      std::vector<double> sad_1_ScalarValues_1;
      std::vector<double> sad_2_ScalarValues_1;
      std::vector<double> minimaScalarValues_1;

      int nb_row_max = maximaScalarValues_1.size();
      int nb_row_sad_1 = sad_1_ScalarValues_1.size();
      int nb_row_sad_2 = sad_2_ScalarValues_1.size(); 
      int nb_row_min = minimaScalarValues_1.size();

      std::vector<std::array<float, 3 >> maximaCoords_2; 
      std::vector<std::array<float, 3 >> minimaCoords_2;
      std::vector<std::array<float, 3 >> sad_1_Coords_2; 
      std::vector<std::array<float, 3 >> sad_2_Coords_2;
      std::vector<double> maximaScalarValues_2;
      std::vector<double> minimaScalarValues_2;
      std::vector<double> sad_1_ScalarValues_2;
      std::vector<double> sad_2_ScalarValues_2;

      int nb_col_max = maximaScalarValues_2.size();
      int nb_col_sad_1 = sad_1_ScalarValues_2.size();
      int nb_col_sad_2 = sad_2_ScalarValues_2.size();
      int nb_col_min = minimaScalarValues_2.size();

      maxMatrix.resize(nb_row_max);
      for (int i = 0 ; i < nb_row_max ; i++){
        maxMatrix[i].resize(nb_col_max);
      }
      sad_1_Matrix.resize(nb_row_sad_1);
      for (int i = 0 ; i < nb_row_sad_1 ; i++){
        sad_1_Matrix[i].resize(nb_col_sad_1);
      }
      sad_2_Matrix.resize(nb_row_sad_2);
      for (int i = 0 ; i < nb_row_sad_2 ; i++){
        sad_2_Matrix[i].resize(nb_col_sad_2);
      }
      minMatrix.resize(nb_row_min);
      for (int i = 0 ; i < nb_row_min ; i++){
        minMatrix[i].resize(nb_col_min);
      }

      float costDeathBirth = epsilon*std::sqrt(computeScalarBoundingBoxSqrd(d1, d2) + boundingBoxRadiusSqrd);

      buildCostMatrix(maximaCoords_1, maximaScalarValues_1, maximaCoords_2, maximaScalarValues_2, maxMatrix, costDeathBirth);
      buildCostMatrix(sad_1_Coords_1, sad_1_ScalarValues_1, sad_1_Coords_2, sad_1_ScalarValues_2, sad_1_Matrix, costDeathBirth);
      buildCostMatrix(sad_2_Coords_1, sad_2_ScalarValues_1, sad_2_Coords_2, sad_2_ScalarValues_2, sad_2_Matrix, costDeathBirth);
      buildCostMatrix(minimaCoords_1, minimaScalarValues_1, minimaCoords_2, minimaScalarValues_2, minMatrix, costDeathBirth);
  }


  void ttk::CriticalPointTracking::auctionAssignement(
      std::vector<std::vector<double>> &costMatrix,
      std::vector<ttk::MatchingType> &matching)
  {
      ttk::AssignmentAuction<double> solver;
      solver.setInput(costMatrix);
      solver.run(matching);
      solver.clearMatrix();
  }



  void ttk::CriticalPointTracking::performMatchings(
    std::vector<DiagramType> persistenceDiagrams, 
    std::vector<std::vector<MatchingType>> &maximaMatchings,
    std::vector<std::vector<MatchingType>> &sad_1_Matchings,
    std::vector<std::vector<MatchingType>> &sad_2_Matchings,
    std::vector<std::vector<MatchingType>> &minimaMatchings, 
    int fieldNumber)
    {
      for (int i = 0 ; i < fieldNumber-1 ; i++){

        std::vector<std::vector<double>> maxMatrix;
        std::vector<std::vector<double>> sad_1_Matrix;
        std::vector<std::vector<double>> sad_2_Matrix;
        std::vector<std::vector<double>> minMatrix;

        std::vector<SimplexId> maxId_1;
        std::vector<SimplexId> sad_1_Id_1;
        std::vector<SimplexId> sad_2_Id_1;
        std::vector<SimplexId> minId_1;
        sortCriticalPoint(persistenceDiagrams[i], maxId_1, sad_1_Id_1, sad_2_Id_1, minId_1);

        std::vector<SimplexId> maxId_2;
        std::vector<SimplexId> sad_1_Id_2;
        std::vector<SimplexId> sad_2_Id_2;
        std::vector<SimplexId> minId_2;
        sortCriticalPoint(persistenceDiagrams[i+1], maxId_2, sad_1_Id_2, sad_2_Id_2, minId_2);

        buildCostMatrices(maxMatrix, sad_1_Matrix, sad_2_Matrix, minMatrix, persistenceDiagrams[i], persistenceDiagrams[i+1]);

        std::vector<ttk::MatchingType> maxMatching;
        std::vector<ttk::MatchingType> sad_1_Matching;
        std::vector<ttk::MatchingType> sad_2_Matching;
        std::vector<ttk::MatchingType> minMatching;

        auctionAssignement(maxMatrix, maxMatching);
        auctionAssignement(sad_1_Matrix, sad_1_Matching);
        auctionAssignement(sad_2_Matrix, sad_2_Matching);
        auctionAssignement(minMatrix, minMatching);

        maximaMatchings.push_back(maxMatching);
        sad_1_Matchings.push_back(sad_1_Matching);
        sad_1_Matchings.push_back(sad_2_Matching);
        minimaMatchings.push_back(minMatching);

      }
    }
