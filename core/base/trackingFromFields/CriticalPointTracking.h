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
#include <Triangulation.h>
#include <TrackingFromPersistenceDiagrams.h>

namespace ttk {

  class CriticalPointTracking : public TrackingFromPersistenceDiagrams, virtual public Debug {


    private:
      double epsilon{10e-1};
      double meshDiameter{1};
      double tolerance{10e-3};

          
    public:
      CriticalPointTracking(){
      
      }

      void setMeshDiamater(double r){
        meshDiameter=r;
      }

      void setEpsilon(double e){
        epsilon=e;
      }

      void setTolerance(double t){
        tolerance = t;
      }

      double computeBoundingBoxRadius(const DiagramType &d1, const DiagramType &d2){
        double maxScalar = d1[0].birth.sfValue;
        double minScalar = d1[0].birth.sfValue;

        for (unsigned int i = 0 ; i < d1.size(); i++){
          maxScalar = std::max(maxScalar, d1[i].birth.sfValue);
          maxScalar = std::max(maxScalar, d1[i].death.sfValue);
          minScalar = std::min(minScalar, d1[i].birth.sfValue);
          minScalar = std::min(minScalar, d1[i].death.sfValue);
        }

        for (unsigned int i = 0 ; i < d2.size(); i++){
          maxScalar = std::max(maxScalar, d2[i].birth.sfValue);
          maxScalar = std::max(maxScalar, d2[i].death.sfValue);
          minScalar = std::min(minScalar, d2[i].birth.sfValue);
          minScalar = std::min(minScalar, d2[i].death.sfValue);
        }

        return std::sqrt(std::pow(meshDiameter, 2)+ std::pow(maxScalar - minScalar, 2));
      }
      
      void performMatchings(
          const std::vector<DiagramType> persistenceDiagrams, 
          std::vector<std::vector<MatchingType>> &outputMatchings,
          int fieldNumber);
    protected:

      double computeRelevantPersistence(const DiagramType &d1, const DiagramType &d2){
        double persistMax = std::abs(d1[0].persistence()); 
        double persistMin = std::abs(d1[0].persistence()); 
        for (unsigned int i = 1 ; i < d1.size() ; i++){
          persistMax = std::max(persistMax, std::abs(d2[i].persistence()));
          persistMin = std::min(persistMin, std::abs(d2[i].persistence()));
        }
        for (unsigned int i = 0 ; i < d2.size() ; i++ ){
          persistMax = std::max(persistMax, std::abs(d1[i].persistence()));
          persistMin = std::min(persistMin, std::abs(d1[i].persistence()));
        }
        return (this->tolerance*(persistMax - persistMin));
      }

      //Compute L_p distance betweem (p,f(p)) and (q,f(q)) where p and q are critical points

      double criticalPointDistance(
          const std::array<float, 3> coords_p1,
          const double sfValue_p1,
          const std::array<float, 3> coords_p2,
          const double sfValue_p2,
          int p);


      //Sort the critical points by types

      void sortCriticalPoint(
          const DiagramType &d, 
          const int t,
          const double minimumRelevantPersistence,
          std::vector<std::array<float, 3>> &maxCoords,
          std::vector<std::array<float, 3>> &sad_1Coords,
          std::vector<std::array<float, 3>> &sad_2Coords,
          std::vector<std::array<float, 3>> &minCoords,
          std::vector<double> &maxScalar,
          std::vector<double> &sad1Scalar,
          std::vector<double> &sad_2Scalar,
          std::vector<double> &minScalar,
          std::vector<std::vector<SimplexId>> &mapMax,
          std::vector<std::vector<SimplexId>> &mapSad_1,
          std::vector<std::vector<SimplexId>> &mapSad_2,
          std::vector<std::vector<SimplexId>> &mapMin);

      void buildCostMatrix(
          const std::vector<std::array<float, 3>> coords_1,
          const std::vector<double> sfValues_1,
          const std::vector<std::array<float, 3>> coords_2,
          const std::vector<double> sfValues_2,
          std::vector<std::vector<double>> &matrix,
          float costDeathBirth);

 


      void localToGlobalMatching(std::vector<std::vector<MatchingType>> &matchings, 
                                  const std::vector<std::vector<int>> &map);

      void auctionAssignement(
          std::vector<std::vector<double>> &costMatrix,
          std::vector<ttk::MatchingType> &matching);
        
      };
}