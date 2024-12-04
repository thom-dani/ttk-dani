#include <AssignmentAuction.h>
#include <AssignmentMunkres.h>
#include <TrackingFromCriticalPoints.h>
#include <algorithm>
// Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are critical
// points

double ttk::TrackingFromCriticalPoints::criticalPointDistance(
  const std::array<float, 3> coords_p1,
  const double sfValue_p1,
  const std::array<float, 3> coords_p2,
  const double sfValue_p2,
  int p = 2) {
  return std::pow(xWeight * std::pow(coords_p1[0] - coords_p2[0], p)
                    + yWeight * std::pow(coords_p1[1] - coords_p2[1], p)
                    + zWeight * std::pow(coords_p1[2] - coords_p2[2], p)
                    + fWeight * std::pow(sfValue_p1 - sfValue_p2, p),
                  1.0 / p);
}

// Sort the critical points by types

void ttk::TrackingFromCriticalPoints::sortCriticalPoint(
  const DiagramType &d,
  const double minimumRelevantPersistence,
  std::vector<std::array<float, 3>> &maxCoords,
  std::vector<std::array<float, 3>> &sad_1Coords,
  std::vector<std::array<float, 3>> &sad_2Coords,
  std::vector<std::array<float, 3>> &minCoords,
  std::vector<double> &maxScalar,
  std::vector<double> &sad_1Scalar,
  std::vector<double> &sad_2Scalar,
  std::vector<double> &minScalar,
  std::vector<SimplexId> &mapMax,
  std::vector<SimplexId> &mapSad_1,
  std::vector<SimplexId> &mapSad_2,
  std::vector<SimplexId> &mapMin,
  std::vector<double> &maxPersistence,
  std::vector<double> &sad_1_Persistence,
  std::vector<double> &sad_2_Persistence,
  std::vector<double> &minPersistence) {
  for(unsigned int i = 0; i < d.size(); i++) {
    std::array<float, 3> birthCoords = d[i].birth.coords;
    std::array<float, 3> deathCoords = d[i].death.coords;
    SimplexId birthId = d[i].birth.id;
    SimplexId deathId = d[i].death.id; 
    if(std::abs(d[i].persistence()) > minimumRelevantPersistence) {
      switch(d[i].dim) {
        case 0:
          minCoords.push_back(birthCoords);
          minScalar.push_back(d[i].birth.sfValue);
          mapMin.push_back(birthId);
          minPersistence.push_back(d[i].persistence());

          sad_1Coords.push_back(deathCoords);
          sad_1Scalar.push_back(d[i].death.sfValue);
          mapSad_1.push_back(deathId);
          sad_1_Persistence.push_back(d[i].persistence());
          break;
        case 1:
          sad_2Coords.push_back(birthCoords);
          sad_2Scalar.push_back(d[i].birth.sfValue);
          mapSad_2.push_back(birthId);
          sad_2_Persistence.push_back(d[i].persistence());

          maxCoords.push_back(deathCoords);
          maxScalar.push_back(d[i].death.sfValue);
          mapMax.push_back(deathId);
          maxPersistence.push_back(d[i].persistence());
          break;
        case 2:
          sad_1Coords.push_back(birthCoords);
          sad_1Scalar.push_back(d[i].birth.sfValue);
          mapSad_1.push_back(birthId);
          sad_1_Persistence.push_back(d[i].persistence());

          sad_2Coords.push_back(deathCoords);
          sad_2Scalar.push_back(d[i].death.sfValue);
          mapSad_2.push_back(deathId);
          sad_2_Persistence.push_back(d[i].persistence());
          break;
      }
    }
  }
}

void ttk::TrackingFromCriticalPoints::buildCostMatrix(
  const std::vector<std::array<float, 3>> coords_1,
  const std::vector<double> sfValues_1,
  const std::vector<std::array<float, 3>> coords_2,
  const std::vector<double> sfValues_2,
  std::vector<std::vector<double>> &matrix,
  float costDeathBirth) {
  int size_1 = coords_1.size();
  int size_2 = coords_2.size();
  int matrix_size = (size_1 > 0 && size_2 > 0) ? size_1 + size_2 : 0;
  for(int i = 0; i < size_1; i++) {
    for(int j = 0; j < size_2; j++) {
      matrix[i][j] = criticalPointDistance(
        coords_1[i], sfValues_1[i], coords_2[j], sfValues_2[j]);
    }
  }
  if(!adaptiveDeathBirthCost){
    for(int i = size_1; i < matrix_size; i++) {
      for(int j = 0; j < size_2; j++) {
        matrix[i][j] = costDeathBirth;
      }
    }
    for(int i = 0; i < size_1; i++) {
      for(int j = size_2; j < matrix_size; j++) {
        matrix[i][j] = costDeathBirth;
      }
    }
  }
  else if(adaptiveDeathBirthCost){
    for (int j = 0 ; j < size_2 ; j++){
      double c=matrix[0][j];
      for (int i = 1 ; i < size_1 ; i++){
        c=matrix[i][j] < c ? matrix[i][j] : c; 
        }
      for (int i = size_1 ; i < matrix_size ; i++){
        matrix[i][j] = c/(epsilonAdapt);
      }
    }

    for (int i = 0 ; i < size_1 ; i++){
      double d=matrix[i][0];
      for (int j = 1 ; j < size_2 ; j++){
        d=matrix[i][j] < d ? matrix[i][j] : d; 
      }
      for (int j = size_2 ; j < matrix_size ; j++){
        matrix[i][j] = d/(epsilonAdapt);
      }
    }
  }
}

void ttk::TrackingFromCriticalPoints::performMatchings(
  const std::vector<DiagramType> persistenceDiagrams,
  std::vector<std::vector<MatchingType>> &maximaMatchings,
  std::vector<std::vector<MatchingType>> &sad_1_Matchings,
  std::vector<std::vector<MatchingType>> &sad_2_Matchings,
  std::vector<std::vector<MatchingType>> &minimaMatchings,
  std::vector<std::vector<MatchingType>> &maxMatchingsPersistence,
  std::vector<std::vector<MatchingType>> &sad_1_MatchingsPersistence,
  std::vector<std::vector<MatchingType>> &sad_2_MatchingsPersistence,
  std::vector<std::vector<MatchingType>> &minMatchingsPersistence,
  int fieldNumber) {

	std::vector<double> sortRT(fieldNumber);
	std::vector<double> matrixRT(fieldNumber - 1);
	std::vector<double> solveRT(fieldNumber - 1);
	std::vector<double> remapRT(fieldNumber - 1);

	std::vector<std::vector<SimplexId>> maxMap(fieldNumber);
	std::vector<std::vector<SimplexId>> sad_1Map(fieldNumber);
	std::vector<std::vector<SimplexId>> sad_2Map(fieldNumber);
	std::vector<std::vector<SimplexId>> minMap(fieldNumber);

	std::vector<std::vector<std::array<float, 3>>>maxCoords(fieldNumber);
	std::vector<std::vector<std::array<float, 3>>>sad_1Coords(fieldNumber);
	std::vector<std::vector<std::array<float, 3>>>sad_2Coords(fieldNumber);
	std::vector<std::vector<std::array<float, 3>>>minCoords(fieldNumber);
	
	std::vector<std::vector<double>> maxScalar(fieldNumber);
	std::vector<std::vector<double>> sad_1Scalar(fieldNumber);
	std::vector<std::vector<double>> sad_2Scalar(fieldNumber);
	std::vector<std::vector<double>> minScalar(fieldNumber);

  std::vector<std::vector<double>> maxPersistence(fieldNumber);
  std::vector<std::vector<double>> sad_1_Persistence(fieldNumber);
  std::vector<std::vector<double>> sad_2_Persistence(fieldNumber);
  std::vector<std::vector<double>> minPersistence(fieldNumber);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
	for (int i = 0 ; i < fieldNumber; i++){
		Timer tm{};
		double clock = tm.getElapsedTime();
		
		double minimumRelevantPersistence{};
		if( i < fieldNumber - 1){
			minimumRelevantPersistence
				= ttk::TrackingFromCriticalPoints::computeRelevantPersistence(
					persistenceDiagrams[i], persistenceDiagrams[i + 1]);
			}
		if( i == fieldNumber - 1){
			minimumRelevantPersistence
				= ttk::TrackingFromCriticalPoints::computeRelevantPersistence(
					persistenceDiagrams[i - 1], persistenceDiagrams[i]);
		}
    sortCriticalPoint(persistenceDiagrams[i], minimumRelevantPersistence,
				maxCoords[i], sad_1Coords[i], sad_2Coords[i], minCoords[i],
				maxScalar[i], sad_1Scalar[i], sad_2Scalar[i], minScalar[i],
				maxMap[i], sad_1Map[i], sad_2Map[i], minMap[i],
        maxPersistence[i],sad_1_Persistence[i], sad_2_Persistence[i], minPersistence[i]);
		sortRT[i]=tm.getElapsedTime() - clock;
} 

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(int i = 0; i < fieldNumber - 1; i++) {
		
		Timer tm{};
		double clock = tm.getElapsedTime();

    float costDeathBirth
      = epsilonConstant
        * computeBoundingBoxRadius(
          persistenceDiagrams[i], persistenceDiagrams[i + 1]);
    int maxSize = (maxCoords[i].size() > 0 && maxCoords[i+1].size() > 0)
                    ? maxCoords[i].size() + maxCoords[i+1].size()
                    : 0;
    int sad_1Size = (sad_1Coords[i].size() > 0 && sad_1Coords[i+1].size() > 0)
                      ? sad_1Coords[i].size() + sad_1Coords[i+1].size()
                      : 0;
    int sad_2Size = (sad_2Coords[i].size() > 0 && sad_2Coords[i+1].size() > 0)
                      ? sad_2Coords[i].size() + sad_2Coords[i+1].size()
                      : 0;
    int minSize = (minCoords[i].size() > 0 && minCoords[i+1].size() > 0)
                    ? minCoords[i].size() + minCoords[i+1].size()
                    : 0;

    std::vector<std::vector<double>> maxMatrix(
      maxSize, std::vector<double>(maxSize, 0));
    std::vector<std::vector<double>> sad_1Matrix(
      sad_1Size, std::vector<double>(sad_1Size, 0));
    std::vector<std::vector<double>> sad_2Matrix(
      sad_2Size, std::vector<double>(sad_2Size, 0));
    std::vector<std::vector<double>> minMatrix(
      minSize, std::vector<double>(minSize, 0));

    std::vector<MatchingType> maxMatching;
    std::vector<MatchingType> sad_1_Matching;
    std::vector<MatchingType> sad_2_Matching;
    std::vector<MatchingType> minMatching;

    buildCostMatrix(maxCoords[i], maxScalar[i], maxCoords[i+1], maxScalar[i+1],
                    maxMatrix, costDeathBirth);
    buildCostMatrix(sad_1Coords[i], sad_1Scalar[i], sad_1Coords[i+1], sad_1Scalar[i+1],
                    sad_1Matrix, costDeathBirth);
    buildCostMatrix(sad_2Coords[i], sad_2Scalar[i], sad_2Coords[i+1], sad_2Scalar[i+1],
                    sad_2Matrix, costDeathBirth);
    buildCostMatrix(minCoords[i], minScalar[i], minCoords[i+1], minScalar[i+1],
                    minMatrix, costDeathBirth);
		
		matrixRT[i]=tm.getElapsedTime() - clock;
		clock = tm.getElapsedTime();

    assignmentSolver(maxMatrix, maxMatching);
    assignmentSolver(sad_1Matrix, sad_1_Matching);
    assignmentSolver(sad_2Matrix, sad_2_Matching);
    assignmentSolver(minMatrix, minMatching);

		solveRT[i]=tm.getElapsedTime() - clock;
		clock = tm.getElapsedTime();

    std::vector<MatchingType> maxPersistence_i(maxMatching.size());
    std::vector<MatchingType> sad_1_Persistence_i(sad_1_Matching.size());
    std::vector<MatchingType> sad_2_Persistence_i(sad_2_Matching.size());
    std::vector<MatchingType> minPersistence_i(minMatching.size());


    localToGlobalMatching(maxMatching, maxMap[i], maxMap[i+1], maxPersistence[i], maxPersistence[i+1], maxPersistence_i);
    localToGlobalMatching(sad_1_Matching, sad_1Map[i], sad_1Map[i+1], sad_1_Persistence[i], sad_1_Persistence[i+1],  sad_1_Persistence_i);
    localToGlobalMatching(sad_2_Matching, sad_2Map[i], sad_2Map[i+1], sad_2_Persistence[i], sad_2_Persistence[i+1], sad_2_Persistence_i);    
    localToGlobalMatching(minMatching, minMap[i], minMap[i+1], minPersistence[i], minPersistence[i+1], minPersistence_i);
    
    maximaMatchings[i] = maxMatching;
    sad_1_Matchings[i] = sad_1_Matching;
    sad_2_Matchings[i] = sad_2_Matching;
    minimaMatchings[i] = minMatching;

    maxMatchingsPersistence[i] = maxPersistence_i;
    sad_1_MatchingsPersistence[i] = sad_1_Persistence_i;
    sad_2_MatchingsPersistence[i] = sad_2_Persistence_i;
    minMatchingsPersistence[i] = minPersistence_i;

		remapRT[i]=tm.getElapsedTime() - clock;
  }
	double RT_1{}, RT_2{}, RT_3{}, RT_4{};
	for (int i = 0 ; i < fieldNumber -1 ; i++){
		RT_1+=sortRT[i];
		RT_2+=matrixRT[i];
		RT_3+=solveRT[i];
		RT_4+=remapRT[i];
		}
		RT_1+=sortRT[fieldNumber - 1];
	std::cout<<std::fixed<<"SortRT = "<<RT_1
											<<",  BuildCostMatrixRT = "<<RT_2
											<<",  SolveRT = "<<RT_3
											<<", RemapRT = "<<RT_4<<std::endl;

}

void ttk::TrackingFromCriticalPoints::localToGlobalMatching(
  std::vector<MatchingType> &matchings,
  const std::vector<int> &startMap,
  const std::vector<int> &endMap,
  const std::vector<double> &startPersistence,
  const std::vector<double> &endPersistence,
  std::vector<MatchingType> &matchingsPersistence) {
  unsigned int totalMatchingsSize = matchings.size();
  int n_removedElements = 0;
  for(unsigned int j = 0; j < totalMatchingsSize ; j++) {
    MatchingType &current_matching = matchings[j - n_removedElements];
    unsigned int id1 = std::get<0>(current_matching);
    unsigned int id2 = std::get<1>(current_matching);
    if(id1 < startMap.size() && id2 < endMap.size()) {

      std::get<0>(matchings[j - n_removedElements]) = startMap[id1];
      std::get<1>(matchings[j - n_removedElements]) = endMap[id2];

      std::get<0>(matchingsPersistence[j - n_removedElements]) = startPersistence[id1];
      std::get<1>(matchingsPersistence[j - n_removedElements]) = endPersistence[id2];

    } else {
      matchings.erase(matchings.begin() + (j - n_removedElements));
      matchingsPersistence.erase(matchingsPersistence.begin() + (j - n_removedElements));
      n_removedElements++;
    }
  }

}

void ttk::TrackingFromCriticalPoints::performTrackingForOneType(
  int fieldNumber,
  std::vector<std::vector<MatchingType>> &matchings,
  std::vector<std::vector<MatchingType>> &matchingPersistence,
  std::vector<trackingTuple> &trackings,
  std::vector<std::vector<double>> &trackingCosts,
  std::vector<double> &trackingPersistence) {
  std::unordered_map<int, int> previousTrackingsEndsIds;
  std::unordered_map<int, int> sw;
  for(unsigned int i = 0; i < matchings[0].size(); i++) {
    std::vector<SimplexId> chain
      = {std::get<0>(matchings[0][i]), std::get<1>(matchings[0][i])};
    std::tuple<int, int, std::vector<SimplexId>> tt
      = std::make_tuple(0, 1, chain);
    trackings.push_back(tt);
    trackingPersistence.push_back(std::get<0>(matchingPersistence[0][i]) + std::get<1>(matchingPersistence[0][i]));
    std::vector<double> newCostEntry;
    newCostEntry.push_back(std::get<2>(matchings[0][i]));

    trackingCosts.push_back(newCostEntry);

    previousTrackingsEndsIds[std::get<1>(matchings[0][i])]
      = trackings.size() - 1;
  }
  for(int i = 1; i < fieldNumber - 1; i++) {
    for(unsigned int j = 0; j < matchings[i].size(); j++) {
      SimplexId v1 = std::get<0>(matchings[i][j]);
      SimplexId v2 = std::get<1>(matchings[i][j]);
      auto it = previousTrackingsEndsIds.find(v1);
      if(it != previousTrackingsEndsIds.end()) {
        std::get<1>(trackings[it->second])++;
        std::get<2>(trackings[it->second]).push_back(v2);
        trackingPersistence[it->second]+=std::get<1>(matchingPersistence[i][j]);
        sw[v2] = it->second;
        trackingCosts[it->second].push_back(std::get<2>(matchings[i][j]));
        previousTrackingsEndsIds.erase(it);
      } else {
        std::vector<ttk::SimplexId> chain = {v1, v2};
        std::tuple<int, int, std::vector<SimplexId>> tt
          = std::make_tuple(i, i + 1, chain);
        trackings.push_back(tt);
        trackingPersistence.push_back(std::get<0>(matchingPersistence[i][j]) + std::get<1>(matchingPersistence[i][j]));
        std::vector<double> newCostEntry;
        newCostEntry.push_back(std::get<2>(matchings[i][j]));
        trackingCosts.push_back(newCostEntry);
        sw[v2] = trackings.size() - 1;
      }
    }
    previousTrackingsEndsIds = sw;
    sw.clear();
  }
  for (unsigned int i = 0; i < trackings.size(); i++){
    trackingPersistence[i]/=(std::get<1>(trackings[i]) - std::get<0>(trackings[i]) + 1);
  }
}

void ttk::TrackingFromCriticalPoints::performTrackings(
  int fieldNumber,
  std::vector<std::vector<MatchingType>> &maximaMatchings,
  std::vector<std::vector<MatchingType>> &sad_1_Matchings,
  std::vector<std::vector<MatchingType>> &sad_2_Matchings,
  std::vector<std::vector<MatchingType>> &minimaMatchings,
  std::vector<std::vector<MatchingType>> &maxMatchingsPersistence,
  std::vector<std::vector<MatchingType>> &sad_1_MatchingsPersistence,
  std::vector<std::vector<MatchingType>> &sad_2_MatchingsPersistence,
  std::vector<std::vector<MatchingType>> &minMatchingsPersistence,
  std::vector<trackingTuple> &allTrackings,
  std::vector<std::vector<double>> &allTrackingsCosts,
  std::vector<double> &allTrackingsMeanPersistences,
  unsigned int (&typesArrayLimits)[]) {

  std::vector<ttk::trackingTuple> trackingsBaseMax;
  std::vector<ttk::trackingTuple> trackingsBaseSad_1;
  std::vector<ttk::trackingTuple> trackingsBaseSad_2;
  std::vector<ttk::trackingTuple> trackingsBaseMin;

  std::vector<std::vector<double>> maxTrackingCost;
  std::vector<std::vector<double>> sad_1_TrackingCost;
  std::vector<std::vector<double>> sad_2_TrackingCost;
  std::vector<std::vector<double>> minTrackingCost;

  std::vector<double> trackingsPersistenceMax;
  std::vector<double> trackingsPersistenceSad_1;
  std::vector<double> trackingsPersistenceSad_2;
  std::vector<double> trackingsPersistenceMin;

  performTrackingForOneType(fieldNumber, maximaMatchings, maxMatchingsPersistence, trackingsBaseMax, maxTrackingCost, trackingsPersistenceMax);
  allTrackings.insert(
    allTrackings.end(), trackingsBaseMax.begin(), trackingsBaseMax.end());
  allTrackingsMeanPersistences.insert(
    allTrackingsMeanPersistences.end(), trackingsPersistenceMax.begin(), trackingsPersistenceMax.end());
  allTrackingsCosts.insert(
    allTrackingsCosts.end(), maxTrackingCost.begin(), maxTrackingCost.end());
  typesArrayLimits[0] = allTrackings.size();

  performTrackingForOneType(fieldNumber, sad_1_Matchings, sad_1_MatchingsPersistence, trackingsBaseSad_1, sad_1_TrackingCost,  trackingsPersistenceSad_1);
  allTrackings.insert(
    allTrackings.end(), trackingsBaseSad_1.begin(), trackingsBaseSad_1.end());
  allTrackingsMeanPersistences.insert(
    allTrackingsMeanPersistences.end(), trackingsPersistenceSad_1.begin(), trackingsPersistenceSad_1.end());
  allTrackingsCosts.insert(
    allTrackingsCosts.end(), sad_1_TrackingCost.begin(), sad_1_TrackingCost.end());
  typesArrayLimits[1] = allTrackings.size();

  performTrackingForOneType(fieldNumber, sad_2_Matchings, sad_2_MatchingsPersistence, trackingsBaseSad_2,  sad_2_TrackingCost, trackingsPersistenceSad_2);
  allTrackings.insert(
    allTrackings.end(), trackingsBaseSad_2.begin(), trackingsBaseSad_2.end());
  allTrackingsMeanPersistences.insert(
    allTrackingsMeanPersistences.end(), trackingsPersistenceSad_2.begin(), trackingsPersistenceSad_2.end());
  allTrackingsCosts.insert(
    allTrackingsCosts.end(), sad_2_TrackingCost.begin(), sad_2_TrackingCost.end());
  typesArrayLimits[2] = allTrackings.size();

  performTrackingForOneType(fieldNumber, minimaMatchings, minMatchingsPersistence, trackingsBaseMin, minTrackingCost,  trackingsPersistenceMin);
  allTrackings.insert(
    allTrackings.end(), trackingsBaseMin.begin(), trackingsBaseMin.end());
  allTrackingsMeanPersistences.insert(
    allTrackingsMeanPersistences.end(), trackingsPersistenceMin.begin(), trackingsPersistenceMin.end());
  allTrackingsCosts.insert(
    allTrackingsCosts.end(), minTrackingCost.begin(), minTrackingCost.end());
}

void ttk::TrackingFromCriticalPoints::assignmentSolver(
  std::vector<std::vector<double>> &costMatrix,
  std::vector<ttk::MatchingType> &matching) {
  if(costMatrix.size() > 0) {
    if(assignmentMethod == 0) {
      ttk::AssignmentAuction<double> solver;
      solver.setInput(costMatrix);
      solver.run(matching);
      solver.clearMatrix();
    } else if(assignmentMethod == 1) {
      ttk::AssignmentMunkres<double> solver;
      solver.setInput(costMatrix);
      solver.run(matching);
      solver.clearMatrix();
    }
  }
}
