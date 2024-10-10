#include <CriticalPointTracking.h>
#include<AssignmentAuction.h>
#include<AssignmentMunkres.h>
#include<algorithm>
//Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are critical points

double ttk::CriticalPointTracking::criticalPointDistance(
    const std::array<float, 3> coords_p1,
    const double sfValue_p1,
    const std::array<float, 3> coords_p2,
    const double sfValue_p2,
    int p = 2){
    return std::pow(xWeight*std::pow(coords_p1[0] - coords_p2[0], p)
                    + yWeight*std::pow(coords_p1[1] - coords_p2[1], p)
                    + zWeight*std::pow(coords_p1[2] - coords_p2[2], p)
                    + fWeight*std::pow(sfValue_p1 - sfValue_p2,p), 1.0/p);
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
            if(std::abs(d[i].persistence()) > minimumRelevantPersistence){
                int persistencePairType = 0;
                if (d[i].birth.type == CriticalType::Saddle1 && d[i].death.type == CriticalType::Saddle2)persistencePairType=1;
                if (d[i].birth.type == CriticalType::Saddle2 && d[i].death.type == CriticalType::Local_maximum)persistencePairType=2;
                switch(persistencePairType){
                    case 0:
                        minCoords.push_back(birthCoords);
                        minScalar.push_back(d[i].birth.sfValue);
                        mapMin[t].push_back(birthId);
                        sad_1Coords.push_back(deathCoords);
                        sad_1Scalar.push_back(d[i].death.sfValue);
                        mapSad_1[t].push_back(deathId);                      
                        break;
                    case 1:
                        sad_1Coords.push_back(birthCoords);
                        sad_1Scalar.push_back(d[i].birth.sfValue);
                        mapSad_1[t].push_back(birthId);    
                        sad_2Coords.push_back(deathCoords);
                        sad_2Scalar.push_back(d[i].death.sfValue);
                        mapSad_2[t].push_back(deathId);
                        break;                                        
                    case 2:
                        sad_2Coords.push_back(birthCoords);
                        sad_2Scalar.push_back(d[i].birth.sfValue);
                        mapSad_2[t].push_back(birthId);    
                        maxCoords.push_back(deathCoords);
                        maxScalar.push_back(d[i].death.sfValue);
                        mapMax[t].push_back(deathId);
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
            for (int j = 0 ; j < size_2 ; j++){
                matrix[i][j] = costDeathBirth;
            }
        }
        for (int i = 0 ; i < size_1 ; i++){
            for (int j = size_2 ; j < matrix_size ; j++){
                matrix[i][j] = costDeathBirth;
            }
        }
    } 

void ttk::CriticalPointTracking::performMatchings(
    const std::vector<DiagramType> persistenceDiagrams,
    std::vector<std::vector<MatchingType>> &maximaMatchings,
    std::vector<std::vector<MatchingType>> &sad_1_Matchings,
    std::vector<std::vector<MatchingType>> &sad_2_Matchings,
    std::vector<std::vector<MatchingType>> &minimaMatchings, 
    int fieldNumber)
    {
   
    std::vector<std::vector<SimplexId>> mapMax(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapSad_1(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapSad_2(fieldNumber, std::vector<SimplexId>(0));
    std::vector<std::vector<SimplexId>> mapMin(fieldNumber, std::vector<SimplexId>(0));
    
    std::vector<std::array<float, 3>> maxCoords_1;
    std::vector<std::array<float, 3>> sad_1Coords_1;
    std::vector<std::array<float, 3>> sad_2Coords_1;
    std::vector<std::array<float, 3>> minCoords_1;

    std::vector<double> maxScalar_1;
    std::vector<double> sad_1Scalar_1;
    std::vector<double> sad_2Scalar_1;
    std::vector<double> minScalar_1;



    double minimumRelevantPersistence = ttk::CriticalPointTracking::computeRelevantPersistence(persistenceDiagrams[0], persistenceDiagrams[1]);

    sortCriticalPoint(persistenceDiagrams[0], 0, minimumRelevantPersistence,
                           maxCoords_1, sad_1Coords_1, sad_2Coords_1, minCoords_1, 
                           maxScalar_1, sad_1Scalar_1, sad_2Scalar_1, minScalar_1,
                           mapMax, mapSad_1, mapSad_2, mapMin);

//    #ifdef TTK_ENABLE_OPENMP
//    #pragma omp parallel for num_threads(threadNumber_)
//    #endif // TTK_ENABLE_OPENMP
    for (int i = 0 ; i < fieldNumber-1 ; i++){

        minimumRelevantPersistence = ttk::CriticalPointTracking::computeRelevantPersistence(persistenceDiagrams[i], persistenceDiagrams[i+1]);
        std::vector<std::array<float, 3>> maxCoords_2;
        std::vector<std::array<float, 3>> sad_1Coords_2;
        std::vector<std::array<float, 3>> sad_2Coords_2;
        std::vector<std::array<float, 3>> minCoords_2;
        std::vector<double> maxScalar_2;
        std::vector<double> sad_1Scalar_2;
        std::vector<double> sad_2Scalar_2;
        std::vector<double> minScalar_2;
        sortCriticalPoint(persistenceDiagrams[i+1], i+1, minimumRelevantPersistence,
                            maxCoords_2, sad_1Coords_2, sad_2Coords_2, minCoords_2, 
                            maxScalar_2, sad_1Scalar_2, sad_2Scalar_2, minScalar_2,
                            mapMax, mapSad_1, mapSad_2, mapMin);

        float costDeathBirth = epsilon*computeBoundingBoxRadius(persistenceDiagrams[i], persistenceDiagrams[i+1]);                
        int maxSize = (maxCoords_1.size() > 0 && maxCoords_2.size() > 0) ? maxCoords_1.size()+maxCoords_2.size() : 0;
        int sad_1Size = (sad_1Coords_1.size() > 0 && sad_1Coords_2.size() > 0) ? sad_1Coords_1.size()+sad_1Coords_2.size() : 0;
        int sad_2Size = (sad_2Coords_1.size() > 0 && sad_2Coords_2.size() > 0) ? sad_2Coords_1.size()+sad_2Coords_2.size() : 0;
        int minSize = (minCoords_1.size() > 0 && minCoords_2.size() > 0) ? minCoords_1.size()+minCoords_2.size() : 0;

        std::vector<std::vector<double>> maxMatrix(maxSize, std::vector<double>(maxSize, 0));
        std::vector<std::vector<double>> sad_1Matrix(sad_1Size, std::vector<double>(sad_1Size, 0));
        std::vector<std::vector<double>> sad_2Matrix(sad_2Size, std::vector<double>(sad_2Size, 0));
        std::vector<std::vector<double>> minMatrix(minSize, std::vector<double>(minSize, 0));

        std::vector<MatchingType> maxMatching;
        std::vector<MatchingType> sad_1_Matching;
        std::vector<MatchingType> sad_2_Matching;
        std::vector<MatchingType> minMatching;

        buildCostMatrix(maxCoords_1, maxScalar_1, maxCoords_2, maxScalar_2, maxMatrix, costDeathBirth);
        buildCostMatrix(sad_1Coords_1, sad_1Scalar_1, sad_1Coords_2, sad_1Scalar_2, sad_1Matrix, costDeathBirth);
        buildCostMatrix(sad_2Coords_1, sad_2Scalar_1, sad_2Coords_2, sad_2Scalar_2, sad_2Matrix, costDeathBirth);
        buildCostMatrix(minCoords_1, minScalar_1, minCoords_2, minScalar_2, minMatrix, costDeathBirth);
        
        assignmentSolver(maxMatrix, maxMatching);
        assignmentSolver(sad_1Matrix, sad_1_Matching);
        assignmentSolver(sad_2Matrix, sad_2_Matching);
        assignmentSolver(minMatrix, minMatching);

        maximaMatchings[i]=maxMatching;
        sad_1_Matchings[i]=sad_1_Matching;
        sad_2_Matchings[i]=sad_2_Matching;
        minimaMatchings[i]=minMatching;
        
        std::swap(maxCoords_1, maxCoords_2);
        std::swap(sad_1Coords_1, sad_1Coords_2);
        std::swap(sad_2Coords_1, sad_2Coords_2);
        std::swap(minCoords_1, minCoords_2);

        std::swap(maxScalar_1, maxScalar_2);
        std::swap(sad_1Scalar_1, sad_1Scalar_2);
        std::swap(sad_2Scalar_1, sad_2Scalar_2);
        std::swap(minScalar_1, minScalar_2);
    }
    localToGlobalMatching(maximaMatchings, mapMax);
    localToGlobalMatching(sad_1_Matchings, mapSad_1);
    localToGlobalMatching(sad_2_Matchings, mapSad_2);
    localToGlobalMatching(minimaMatchings, mapMin);
    }

    void ttk::CriticalPointTracking::localToGlobalMatching(std::vector<std::vector<MatchingType>> &matchings, 
                                                            const std::vector<std::vector<int>> &map){

        for (unsigned int i = 0 ; i < matchings.size() ; i++){
            unsigned int start_size = matchings[i].size();
            int removedElements =0;
            for (unsigned int j = 0 ; j < start_size ; j++){
                MatchingType &current_matching = matchings[i][j-removedElements];
                unsigned int id1 = std::get<0>(current_matching);
                unsigned int id2 = std::get<1>(current_matching);
                if(id1 < map[i].size() && id2 < map[i+1].size()){
                    std::get<0>(matchings[i][j-removedElements])= map[i][id1]; 
                    std::get<1>(matchings[i][j - removedElements])= map[i+1][id2]; 
                }
                else{
                    matchings[i].erase(matchings[i].begin() + (j-removedElements));
                    removedElements++;
                }
            }
        }
    }

     void ttk::CriticalPointTracking::performTrackingForOneType(
          int fieldNumber, 
          std::vector<std::vector<MatchingType>> &matchings, 
          std::vector<trackingTuple> &trackings){
            std::unordered_map<int, int> previousTrackingsEndsIds;
            std::unordered_map<int, int> sw;
            for (unsigned int i = 0 ; i< matchings[0].size() ; i++){
                std::vector<SimplexId> chain = {std::get<0>(matchings[0][i]), std::get<1>(matchings[0][i])};
                std::tuple<int, int, std::vector<SimplexId>> tt = std::make_tuple(0, 1, chain);
                trackings.push_back(tt);
                previousTrackingsEndsIds[std::get<1>(matchings[0][i])] = trackings.size()-1;
            }
            for (int i = 1  ; i < fieldNumber-1 ; i++){
                for (unsigned int j = 0 ; j < matchings[i].size() ; j++){
                    SimplexId v1 = std::get<0>(matchings[i][j]);
                    SimplexId v2 = std::get<1>(matchings[i][j]);
                    auto it = previousTrackingsEndsIds.find(v1);
                    if (it != previousTrackingsEndsIds.end()){
                        std::get<1>(trackings[it->second])++;
                        std::get<2>(trackings[it->second]).push_back(v2);
                        sw[v2]=it->second;
                    }
                    else{
                        std::vector<ttk::SimplexId> chain = {v1, v2};
                        std::tuple<int, int, std::vector<SimplexId>> tt = std::make_tuple(i, i+1, chain);
                        trackings.push_back(tt);
                        sw[v2]=trackings.size()-1;
                    }
                }
                previousTrackingsEndsIds=sw;
                sw.clear();
            }
        }

    void ttk::CriticalPointTracking::performTrackings(
        const std::vector<DiagramType> persistenceDiagrams,
        std::vector<std::vector<MatchingType>> &maximaMatchings,
        std::vector<std::vector<MatchingType>> &sad_1_Matchings,
        std::vector<std::vector<MatchingType>> &sad_2_Matchings,
        std::vector<std::vector<MatchingType>> &minimaMatchings,
        std::vector<trackingTuple> &allTrackings,
        unsigned int  (&typesArrayLimits)[]
      ){

        int fieldNumber = persistenceDiagrams.size();
        std::vector<ttk::trackingTuple> trackingsBaseMax;
        std::vector<ttk::trackingTuple> trackingsBaseSad_1;
        std::vector<ttk::trackingTuple> trackingsBaseSad_2;
        std::vector<ttk::trackingTuple> trackingsBaseMin;

        performTrackingForOneType(fieldNumber, maximaMatchings, trackingsBaseMax);
        allTrackings.insert(allTrackings.end(), trackingsBaseMax.begin(), trackingsBaseMax.end());   
        typesArrayLimits[0]=allTrackings.size();

        performTrackingForOneType(fieldNumber, sad_1_Matchings, trackingsBaseSad_1);
        allTrackings.insert(allTrackings.end(), trackingsBaseSad_1.begin(), trackingsBaseSad_1.end());
        typesArrayLimits[1]=allTrackings.size();

        performTrackingForOneType(fieldNumber, sad_2_Matchings, trackingsBaseSad_2);
        allTrackings.insert(allTrackings.end(), trackingsBaseSad_2.begin(), trackingsBaseSad_2.end());   
        typesArrayLimits[2]=allTrackings.size();

        performTrackingForOneType(fieldNumber, minimaMatchings, trackingsBaseMin);
        allTrackings.insert(allTrackings.end(), trackingsBaseMin.begin(), trackingsBaseMin.end());  
      }

    void ttk::CriticalPointTracking::assignmentSolver(
      std::vector<std::vector<double>> &costMatrix,
      std::vector<ttk::MatchingType> &matching){
        if(assignmentMethod == 0){
            ttk::AssignmentAuction<double> solver;
            solver.setInput(costMatrix);
            solver.setNumberOfRounds(100);
            solver.setEpsilon(10e-1);
            solver.run(matching);
            solver.clearMatrix();
        }
        else if(assignmentMethod == 1){
            ttk::AssignmentMunkres<double> solver;
            solver.setInput(costMatrix);
            solver.run(matching);
            solver.clearMatrix();
        }
    }


