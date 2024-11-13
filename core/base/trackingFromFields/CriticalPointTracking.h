/// \ingroup base
/// \class ttk::TrackingFromPersistenceDiagrams
/// \author Thomas Daniel <thomas.daniel@lip6.fr>
/// \date Septemeber 2024.
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/timeTracking/">Time
///   tracking example</a>

#pragma once

// base code includes
#include <PersistenceDiagram.h>
#include <TrackingFromPersistenceDiagrams.h>
#include <Triangulation.h>

namespace ttk {

  class CriticalPointTracking : virtual public Debug,
                                public TrackingFromPersistenceDiagrams {

  private:
    double epsilon{10e-1};
    double meshDiameter{1};
    double tolerance{10e-3};
    int assignmentMethod{0};
    double xWeight{1};
    double yWeight{1};
    double zWeight{1};
    double fWeight{1};
    bool adaptiveDeathBirthCost{false};

  public:
    CriticalPointTracking() {
    }

    void setMeshDiamater(double r) {
      meshDiameter = r;
    }

    void setEpsilon(double e) {
      epsilon = e;
    }

    void setTolerance(double t) {
      tolerance = t;
    }

    void setAssignmentMethod(int a) {
      if(a == 0 || a == 1) {
        assignmentMethod = a;
      }
    }
    
    void setAdaptDeathBirthCost(bool b){
      adaptiveDeathBirthCost=b;
    }


    void setWeights(double PX, double PY, double PZ, double PF) {
      xWeight = PX;
      yWeight = PY;
      zWeight = PZ;
      fWeight = PF;
    }

    double computeBoundingBoxRadius(const DiagramType &d1,
                                    const DiagramType &d2) {
      double maxScalar = d1[0].birth.sfValue;
      double minScalar = d1[0].birth.sfValue;

      for(unsigned int i = 0; i < d1.size(); i++) {
        maxScalar = std::max(maxScalar, d1[i].birth.sfValue);
        maxScalar = std::max(maxScalar, d1[i].death.sfValue);
        minScalar = std::min(minScalar, d1[i].birth.sfValue);
        minScalar = std::min(minScalar, d1[i].death.sfValue);
      }

      for(unsigned int i = 0; i < d2.size(); i++) {
        maxScalar = std::max(maxScalar, d2[i].birth.sfValue);
        maxScalar = std::max(maxScalar, d2[i].death.sfValue);
        minScalar = std::min(minScalar, d2[i].birth.sfValue);
        minScalar = std::min(minScalar, d2[i].death.sfValue);
      }

      return std::sqrt(std::pow(meshDiameter, 2)
                       + std::pow(maxScalar - minScalar, 2));
    }

    void
      performMatchings(const std::vector<DiagramType> persistenceDiagrams,
                       std::vector<std::vector<MatchingType>> &maximaMatchings,
                       std::vector<std::vector<MatchingType>> &sad_1_Matchings,
                       std::vector<std::vector<MatchingType>> &sad_2_Matchings,
                       std::vector<std::vector<MatchingType>> &minimaMatchings,
                       std::vector<std::vector<MatchingType>> &maxMatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &sad_1_MatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &sad_2_MatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &minMatchingsPersistence,
                       int fieldNumber);

    void
      performTrackings(int fieldNumber,
                       std::vector<std::vector<MatchingType>> &maximaMatchings,
                       std::vector<std::vector<MatchingType>> &sad_1_Matchings,
                       std::vector<std::vector<MatchingType>> &sad_2_Matchings,
                       std::vector<std::vector<MatchingType>> &minimaMatchings,
                       std::vector<std::vector<MatchingType>> &maxMatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &sad_1_MatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &sad_2_MatchingsPersistence,
                       std::vector<std::vector<MatchingType>> &minMatchingsPersistence,
                       std::vector<trackingTuple> &allTrackings,
                       std::vector<std::vector<double>> &allTrackingCost,
                       std::vector<double> &allTrackingsMeanPersistences,
                       unsigned int (&sizes)[]);

  protected:
    double computeRelevantPersistence(const DiagramType &d1,
                                      const DiagramType &d2) {
      const auto sp = this->tolerance;
      const double s = sp > 0.0 && sp < 100.0 ? sp / 100.0 : 0;

      std::vector<double> toSort(d1.size() + d2.size());
      for(size_t i = 0; i < d1.size(); ++i) {
        const auto &t = d1[i];
        toSort[i] = std::abs(t.persistence());
      }
      for(size_t i = 0; i < d2.size(); ++i) {
        const auto &t = d2[i];
        toSort[d1.size() + i] = std::abs(t.persistence());
      }

      const auto minVal = *std::min_element(toSort.begin(), toSort.end());
      const auto maxVal = *std::max_element(toSort.begin(), toSort.end());
      return s * (maxVal - minVal);
    }

    // Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are
    // critical points

    double criticalPointDistance(const std::array<float, 3> coords_p1,
                                 const double sfValue_p1,
                                 const std::array<float, 3> coords_p2,
                                 const double sfValue_p2,
                                 int p);

    // Sort the critical points by types

    void sortCriticalPoint(const DiagramType &d,
                           const double minimumRelevantPersistence,
                           std::vector<std::array<float, 3>> &maxCoords,
                           std::vector<std::array<float, 3>> &sad_1Coords,
                           std::vector<std::array<float, 3>> &sad_2Coords,
                           std::vector<std::array<float, 3>> &minCoords,
                           std::vector<double> &maxScalar,
                           std::vector<double> &sad1Scalar,
                           std::vector<double> &sad_2Scalar,
                           std::vector<double> &minScalar,
                           std::vector<SimplexId> &mapMax,
                           std::vector<SimplexId> &mapSad_1,
                           std::vector<SimplexId> &mapSad_2,
                           std::vector<SimplexId> &mapMin,
                           std::vector<double> &maxPersistence,
                           std::vector<double> &sad_1_Persistence,
                           std::vector<double> &sad_2_Persistence,
                           std::vector<double> &minPersistence);

    void buildCostMatrix(const std::vector<std::array<float, 3>> coords_1,
                         const std::vector<double> sfValues_1,
                         const std::vector<std::array<float, 3>> coords_2,
                         const std::vector<double> sfValues_2,
                         std::vector<std::vector<double>> &matrix,
                         float costDeathBirth);

    void localToGlobalMatching(std::vector<MatchingType> &matchings,
                               const std::vector<int> &startMap,
                               const std::vector<int> &endMap,
                               const std::vector<double> &startPersistence,
                               const std::vector<double> &endPersistence,
                               std::vector<MatchingType> &matchingsPersistence);

    void assignmentSolver(std::vector<std::vector<double>> &costMatrix,
                          std::vector<ttk::MatchingType> &matching);

    void performTrackingForOneType(
      int fieldNumber,
      std::vector<std::vector<MatchingType>> &matching,
      std::vector<std::vector<MatchingType>> &maxMatchingsPersistence,
      std::vector<trackingTuple> &tracking,
      std::vector<std::vector<double>> &trackingCosts,
      std::vector<double> &trackingPersistence);
  };
} // namespace ttk