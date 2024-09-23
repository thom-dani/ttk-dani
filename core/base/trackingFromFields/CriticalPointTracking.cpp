#include <CriticalPointTracking.h>
#include<AssignmentAuction.h>
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
    std::vector<std::array<float, 3>> &maxCoords,
    std::vector<std::array<float, 3>> &sad_1Coords,
    std::vector<std::array<float, 3>> &sad_2Coords,
    std::vector<std::array<float, 3>> &minCoords,
    std::vector<double> &maxScalar,
    std::vector<double> &sad_1Scalar,
    std::vector<double> &sad_2Scalar,
    std::vector<double> &minScalar)
    {
        for (unsigned int i = 0 ; i < d.size() ; i++ ){
          std::array<float,3> birthCoords = d[i].birth.coords;
          std::array<float,3> deathCoords = d[i].death.coords;
            switch(d[i].birth.type){
                case CriticalType::Local_maximum:
                    maxCoords.push_back(birthCoords);
                    maxScalar.push_back(d[i].birth.sfValue);
                case CriticalType::Saddle1:
                    sad_1Coords.push_back(birthCoords);
                    sad_1Scalar.push_back(d[i].birth.sfValue);
                case CriticalType::Saddle2:
                    sad_2Coords.push_back(birthCoords);
                    sad_2Scalar.push_back(d[i].birth.sfValue);
                case CriticalType::Local_minimum:
                    minCoords.push_back(birthCoords);
                    minScalar.push_back(d[i].birth.sfValue);
                default : 
                    break;
            }
            switch(d[i].death.type){
                case CriticalType::Local_maximum:
                    maxCoords.push_back(deathCoords);
                    maxScalar.push_back(d[i].death.sfValue);
                case CriticalType::Saddle1:
                    sad_1Coords.push_back(deathCoords);
                    sad_1Scalar.push_back(d[i].death.sfValue);
                case CriticalType::Saddle2:
                    sad_2Coords.push_back(deathCoords);
                    sad_2Scalar.push_back(d[i].death.sfValue);
                case CriticalType::Local_minimum:
                    minCoords.push_back(deathCoords);
                    minScalar.push_back(d[i].death.sfValue);
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

void ttk::CriticalPointTracking::performMatchings(
    std::vector<DiagramType> persistenceDiagrams, 
    std::vector<std::vector<MatchingType>> &maximaMatchings,
    std::vector<std::vector<MatchingType>> &sad_1_Matchings,
    std::vector<std::vector<MatchingType>> &sad_2_Matchings,
    std::vector<std::vector<MatchingType>> &minimaMatchings, 
    int fieldNumber)
    {

    std::vector<std::vector<int>> mapMax(fieldNumber);
    std::vector<std::vector<int>> mapSad_1(fieldNumber);
    std::vector<std::vector<int>> mapSad_2(fieldNumber);
    std::vector<std::vector<int>> mapMin(fieldNumber);
   
    for (int i = 0 ; i < fieldNumber-1 ; i++){

        std::vector<std::array<float, 3>> maxCoords_1;
        std::vector<std::array<float, 3>> sad_1Coords_1;
        std::vector<std::array<float, 3>> sad_2Coords_1;
        std::vector<std::array<float, 3>> minCoords_1;

        std::vector<double> maxScalar_1;
        std::vector<double> sad_1Scalar_1;
        std::vector<double> sad_2Scalar_1;
        std::vector<double> minScalar_1;

        sortCriticalPoint(persistenceDiagrams[i], 
                            maxCoords_1, sad_1Coords_1, sad_2Coords_1, minCoords_1, 
                            maxScalar_1, sad_1Scalar_1, sad_2Scalar_1, minScalar_1,
                            mapMax, mapSad_1, mapSad_2, mapMin);

        std::vector<std::array<float, 3>> maxCoords_2;
        std::vector<std::array<float, 3>> sad_1Coords_2;
        std::vector<std::array<float, 3>> sad_2Coords_2;
        std::vector<std::array<float, 3>> minCoords_2;

        std::vector<double> maxScalar_2;
        std::vector<double> sad_1Scalar_2;
        std::vector<double> sad_2Scalar_2;
        std::vector<double> minScalar_2;

        sortCriticalPoint(persistenceDiagrams[i+1], 
                            maxCoords_2, sad_1Coords_2, sad_2Coords_2, minCoords_2, 
                          maxScalar_2, sad_1Scalar_2, sad_2Scalar_2, minScalar_2);
        
        float costDeathBirth = computeBoundingBoxRadius(persistenceDiagrams[i], persistenceDiagrams[i+1]);
        
        int maxSize = maxCoords_1.size()+maxCoords_2.size();
        int sad_1Size = sad_2Coords_1.size()+sad_1Coords_2.size();
        int sad_2Size = sad_2Coords_1.size()+sad_2Coords_2.size();
        int minSize = minCoords_1.size()+minCoords_2.size();

        std::vector<std::vector<double>> maxMatrix(maxSize, std::vector<double>(maxSize, 0));
        std::vector<std::vector<double>> sad_1Matrix(sad_1Size, std::vector<double>(sad_1Size, 0));
        std::vector<std::vector<double>> sad_2Matrix(sad_2Size, std::vector<double>(sad_2Size, 0));
        std::vector<std::vector<double>> minMatrix(minSize, std::vector<double>(minSize, 0));

        buildCostMatrix(maxCoords_1, maxScalar_1, maxCoords_2, maxScalar_2, maxMatrix, costDeathBirth);
        buildCostMatrix(sad_1Coords_1, sad_1Scalar_1, maxCoords_2, maxScalar_2, sad_1Matrix, costDeathBirth);
        buildCostMatrix(sad_2Coords_1, maxScalar_1, maxCoords_2, maxScalar_2, sad_2Matrix, costDeathBirth);
        buildCostMatrix(maxCoords_1, maxScalar_1, maxCoords_2, maxScalar_2, minMatrix, costDeathBirth);

        std::vector<ttk::MatchingType> maxMatching;
        std::vector<ttk::MatchingType> sad_1_Matching;
        std::vector<ttk::MatchingType> sad_2_Matching;
        std::vector<ttk::MatchingType> minMatching;

        auctionAssignement(maxMatrix, maxMatching);
        auctionAssignement(sad_1Matrix, sad_1_Matching);
        auctionAssignement(sad_2Matrix, sad_2_Matching);
        auctionAssignement(minMatrix, minMatching);

        maximaMatchings.push_back(maxMatching);
        sad_1_Matchings.push_back(sad_1_Matching);
        sad_1_Matchings.push_back(sad_2_Matching);
        minimaMatchings.push_back(minMatching);

      }
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

