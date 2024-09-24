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
    const int t,
    const double minimumRelevantPersistence,
    std::vector<std::array<float, 3>> &maxCoords,
    std::vector<std::array<float, 3>> &sad_1Coords,
    std::vector<std::array<float, 3>> &sad_2Coords,
    std::vector<std::array<float, 3>> &minCoords,
    std::vector<double> &maxScalar,
    std::vector<double> &sad_1Scalar,
    std::vector<double> &sad_2Scalar,
    std::vector<double> &minScalar,
    std::vector<std::vector<SimplexId>> &mapMax,
    std::vector<std::vector<SimplexId>> &mapSad_1,
    std::vector<std::vector<SimplexId>> &mapSad_2,
    std::vector<std::vector<SimplexId>> &mapMin)
    {
        for (unsigned int i = 0 ; i < d.size() ; i++ ){
            std::array<float,3> birthCoords = d[i].birth.coords;
            std::array<float,3> deathCoords = d[i].death.coords;
            SimplexId birthId = d[i].birth.id;
            SimplexId deathId = d[i].death.id;
            if(d[i].persistence() > minimumRelevantPersistence){
                switch(d[i].birth.type){
                    case CriticalType::Local_maximum:
                        maxCoords.push_back(birthCoords);
                        maxScalar.push_back(d[i].birth.sfValue);
                        mapMax[t].push_back(birthId);
                        break;
                    case CriticalType::Saddle1:
                        sad_1Coords.push_back(birthCoords);
                        sad_1Scalar.push_back(d[i].birth.sfValue);
                        mapSad_1[t].push_back(birthId);
                        break;                    
                    case CriticalType::Saddle2:
                        sad_2Coords.push_back(birthCoords);
                        sad_2Scalar.push_back(d[i].birth.sfValue);
                        mapSad_2[t].push_back(birthId);
                        break;
                    case CriticalType::Local_minimum:
                        minCoords.push_back(birthCoords);
                        minScalar.push_back(d[i].birth.sfValue);
                        mapMin[t].push_back(birthId);
                        break;
                    default : 
                        break;
                }
                switch(d[i].death.type){
                    case CriticalType::Local_maximum:
                        maxCoords.push_back(deathCoords);
                        maxScalar.push_back(d[i].death.sfValue);
                        mapMax[t].push_back(deathId);
                        break;
                    case CriticalType::Saddle1:
                        sad_1Coords.push_back(deathCoords);
                        sad_1Scalar.push_back(d[i].death.sfValue);
                        mapSad_1[t].push_back(deathId);
                        break;
                    case CriticalType::Saddle2:
                        sad_2Coords.push_back(deathCoords);
                        sad_2Scalar.push_back(d[i].death.sfValue);
                        mapSad_2[t].push_back(deathId);
                        break;
                    case CriticalType::Local_minimum:
                        minCoords.push_back(deathCoords);
                        minScalar.push_back(d[i].death.sfValue);
                        mapMin[t].push_back(deathId);
                        break;
                    default : 
                        break;
                }
            }
        }  
    }


void ttk::CriticalPointTracking::buildCostMatrix(
    const std::vector<std::array<float, 3>> coords_1,
    const std::vector<double> sfValues_1,
    const std::vector<std::array<float, 3>> coords_2,
    const std::vector<double> sfValues_2,
    std::vector<std::vector<double>> &matrix,
    float costDeathBirth)
    {
        int size_1 = coords_1.size();
        int size_2 = coords_2.size();
        int matrix_size = size_1 + size_2;
        for (int i = 0 ; i < size_1 ; i++){
            for (int j = 0 ; j < size_2 ; j++){
                matrix[i][j]=criticalPointDistance(coords_1[i], sfValues_1[i], coords_2[j], sfValues_2[j]);
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
    const std::vector<DiagramType> persistenceDiagrams,
    std::vector<std::vector<MatchingType>> &outuputMatchings,
    int fieldNumber)
    {
    std::vector<std::vector<MatchingType>> maximaMatchings;
    std::vector<std::vector<MatchingType>> sad_1_Matchings;
    std::vector<std::vector<MatchingType>> sad_2_Matchings;
    std::vector<std::vector<MatchingType>> minimaMatchings; 
    std::vector<std::vector<SimplexId>> mapMax(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapSad_1(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapSad_2(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapMin(fieldNumber, std::vector<SimplexId>(0));
    
    std::vector<std::array<float, 3>> maxCoords_1(0);
    std::vector<std::array<float, 3>> sad_1Coords_1(0);
    std::vector<std::array<float, 3>> sad_2Coords_1(0);
    std::vector<std::array<float, 3>> minCoords_1(0);

    std::vector<double> maxScalar_1(0);
    std::vector<double> sad_1Scalar_1(0);
    std::vector<double> sad_2Scalar_1(0);
    std::vector<double> minScalar_1(0);

    double minimumRelevantPersistence = ttk::CriticalPointTracking::computeRelevantPersistence(persistenceDiagrams[0], persistenceDiagrams[1]);

    sortCriticalPoint(persistenceDiagrams[0], 0, minimumRelevantPersistence,
                           maxCoords_1, sad_1Coords_1, sad_2Coords_1, minCoords_1, 
                           maxScalar_1, sad_1Scalar_1, sad_2Scalar_1, minScalar_1,
                           mapMax, mapSad_1, mapSad_2, mapMin);

   
    for (int i = 0 ; i < fieldNumber-1 ; i++){

        minimumRelevantPersistence = ttk::CriticalPointTracking::computeRelevantPersistence(persistenceDiagrams[i], persistenceDiagrams[i+1]);

        std::vector<std::array<float, 3>> maxCoords_2(0);
        std::vector<std::array<float, 3>> sad_1Coords_2(0);
        std::vector<std::array<float, 3>> sad_2Coords_2(0);
        std::vector<std::array<float, 3>> minCoords_2(0);

        std::vector<double> maxScalar_2(0);
        std::vector<double> sad_1Scalar_2(0);
        std::vector<double> sad_2Scalar_2(0);
        std::vector<double> minScalar_2(0);

        sortCriticalPoint(persistenceDiagrams[i+1], i+1, minimumRelevantPersistence,
                            maxCoords_2, sad_1Coords_2, sad_2Coords_2, minCoords_2, 
                            maxScalar_2, sad_1Scalar_2, sad_2Scalar_2, minScalar_2,
                            mapMax, mapSad_1, mapSad_2, mapMin);
        
        float costDeathBirth = computeBoundingBoxRadius(persistenceDiagrams[i], persistenceDiagrams[i+1]);                
        int maxSize = maxCoords_1.size()+maxCoords_2.size();
        int sad_1Size = sad_1Coords_1.size()+sad_1Coords_2.size();
        int sad_2Size = sad_2Coords_1.size()+sad_2Coords_2.size();
        int minSize = minCoords_1.size()+minCoords_2.size();

        std::vector<std::vector<double>> maxMatrix(maxSize, std::vector<double>(maxSize, 0));
        std::vector<std::vector<double>> sad_1Matrix(sad_1Size, std::vector<double>(sad_1Size, 0));
        std::vector<std::vector<double>> sad_2Matrix(sad_2Size, std::vector<double>(sad_2Size, 0));
        std::vector<std::vector<double>> minMatrix(minSize, std::vector<double>(minSize, 0));
   
        buildCostMatrix(maxCoords_1, maxScalar_1, maxCoords_2, maxScalar_2, maxMatrix, costDeathBirth);
        buildCostMatrix(sad_1Coords_1, sad_1Scalar_1, sad_1Coords_2, sad_1Scalar_2, sad_1Matrix, costDeathBirth);
        buildCostMatrix(sad_2Coords_1, sad_2Scalar_1, sad_2Coords_2, sad_2Scalar_2, sad_2Matrix, costDeathBirth);
        buildCostMatrix(minCoords_1, minScalar_1, minCoords_2, minScalar_2, minMatrix, costDeathBirth);

        std::vector<MatchingType> maxMatching;
        std::vector<MatchingType> sad_1_Matching;
        std::vector<MatchingType> sad_2_Matching;
        std::vector<MatchingType> minMatching;
        if(maxSize > 0)auctionAssignement(maxMatrix, maxMatching);
        if(sad_1Size > 0)auctionAssignement(sad_1Matrix, sad_1_Matching);
        if(sad_2Size > 0)auctionAssignement(sad_2Matrix, sad_2_Matching);
        if(minSize > 0)auctionAssignement(minMatrix, minMatching);

        maximaMatchings.push_back(maxMatching);
        sad_1_Matchings.push_back(sad_1_Matching);
        sad_2_Matchings.push_back(sad_2_Matching);
        minimaMatchings.push_back(minMatching);
        
        maxCoords_1 = maxCoords_2;
        sad_1Coords_1 = sad_1Coords_2;
        sad_2Coords_1 = sad_2Coords_2;
        minCoords_1 =minCoords_2;

        maxScalar_1 = maxScalar_2;
        sad_1Scalar_1 = sad_1Scalar_2;
        sad_2Scalar_1 = sad_2Scalar_2;
        minScalar_1 = minScalar_2;
    }


    localToGlobalMatching(maximaMatchings, mapMax);
    localToGlobalMatching(sad_1_Matchings, mapSad_1);
    localToGlobalMatching(sad_2_Matchings, mapSad_2);
    localToGlobalMatching(minimaMatchings, mapMin);

   

    for (int i = 0 ; i < fieldNumber - 1 ; i++){
        std::vector<MatchingType> matching_i;
        matching_i.insert(matching_i.end(), maximaMatchings[i].begin() , maximaMatchings[i].end());
        matching_i.insert(matching_i.end(), sad_1_Matchings[i].begin() , sad_1_Matchings[i].end());
        matching_i.insert(matching_i.end(), sad_2_Matchings[i].begin() , sad_2_Matchings[i].end());
        matching_i.insert(matching_i.end(), minimaMatchings[i].begin() , minimaMatchings[i].end());
        outuputMatchings.push_back(matching_i);
    }

    }

    void ttk::CriticalPointTracking::localToGlobalMatching(std::vector<std::vector<MatchingType>> &matchings, 
                                                            const std::vector<std::vector<int>> &map){

        for (unsigned int i = 0 ; i < matchings.size() ; i++){
            for (unsigned int j = 0 ; j < matchings[i].size() ; j++){
                MatchingType current_matching = matchings[i][j];
                unsigned int id1 = std::get<0>(current_matching);
                unsigned int id2 = std::get<1>(current_matching);
                SimplexId globalId1 = id1 >= map[i].size() ? -1 : map[i][id1];
                SimplexId globalId2 = id2 >= map[i].size() ? -1 : map[i+1][id2];;

                std::get<0>(current_matching)=globalId1; 
                std::get<1>(current_matching)=globalId2; 
            }
        }
    }


  void ttk::CriticalPointTracking::auctionAssignement(
      std::vector<std::vector<double>> &costMatrix,
      std::vector<ttk::MatchingType> &matching)
    {
        ttk::AssignmentAuction<double> solver;
        solver.setInput(costMatrix);
        solver.setNumberOfRounds(100);
        solver.setEpsilon(10e-1);
        solver.setEpsilonDiviserMultiplier(1);
        solver.run(matching);
        solver.clearMatrix();
    }

