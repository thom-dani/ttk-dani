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

namespace ttk {

  class CriticalPointTracking : virtual public Debug {


    private:
      double epsilon;
      double meshDiameter;
          
    public:
      CriticalPointTracking(){
      }

      void setBoudingBoxRadius(double r){
        meshDiameter=r;
      }

      void setBoudingBoxRadius(double e){
        epsilon=e;
      }

      double computeBoundingBoxRadius(const DiagramType &d1, const DiagramType &d2){
        double maxScalar, minScalar;

        for (int i = 0 ; i < d1.size(); i++){
          maxScalar = std::max(maxScalar, d1[i].birth.sfValue);
          maxScalar = std::max(maxScalar, d1[i].death.sfValue);
          minScalar = std::min(minScalar, d1[i].birth.sfValue);
          minScalar = std::min(minScalar, d1[i].death.sfValue);
        }

        for (int i = 0 ; i < d2.size(); i++){
          maxScalar = std::max(maxScalar, d2[i].birth.sfValue);
          maxScalar = std::max(maxScalar, d2[i].death.sfValue);
          minScalar = std::min(minScalar, d2[i].birth.sfValue);
          minScalar = std::min(minScalar, d2[i].death.sfValue);
        }

        return std::sqrt(std::pow(meshDiameter, 2)+ std::pow(maxScalar - minScalar, 2));
      }
    protected:

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
          std::vector<std::array<float, 3>> &maxCoords,
          std::vector<std::array<float, 3>> &sad_1Coords,
          std::vector<std::array<float, 3>> &sad_2Coords,
          std::vector<std::array<float, 3>> &minCoords,
          std::vector<double> &maxScalar,
          std::vector<double> &sad1Scalar,
          std::vector<double> &sad_2Scalar,
          std::vector<double> &minScalar);

      void buildCostMatrix(
          const std::vector<std::array<float, 3>> coords_1,
          const std::vector<double> sfValues_1,
          const std::vector<std::array<float, 3>> coords_2,
          const std::vector<double> sfValues_2,
          std::vector<std::vector<double>> &matrix,
          float costDeathBirth);
       
      void buildCostMatrices(
        const std::vector<std::array<float, 3>> pointSetCoords_1,
        const std::vector<double> pointSetScalar_1,
        const std::vector<std::array<float, 3>> pointSetCoords_2,
        const std::vector<double> pointSetScalar_2,
        std::vector<std::vector<double>> &costMatrice);

      void auctionAssignement(
          std::vector<std::vector<double>> &costMatrix,
          std::vector<ttk::MatchingType> &matching);
        


      void performMatchings(
        std::vector<DiagramType> persistenceDiagrams, 
        std::vector<std::vector<MatchingType>> &maximaMatchings,
        std::vector<std::vector<MatchingType>> &sad_1_Matchings,
        std::vector<std::vector<MatchingType>> &sad_2_Matchings,
        std::vector<std::vector<MatchingType>> &minimaMatchings, 
        int fieldNumber);
         

      void buildCostMatrix(
        const std::vector<std::array<float, 3>> coords_1,
        const std::vector<double> sfValues_1,
        const std::vector<std::array<float, 3>> coords_2,
        const std::vector<double> sfValues_2,
        std::vector<std::vector<double>> &matrix,
        float costDeathBirth
        );

      void buildCostMatrices(
        const std::vector<SimplexId> &idDiagram_1,
        const std::vector<SimplexId> &idDiagram_2,
        const DiagramType &d1,
        const DiagramType &d2,
        std::vector<std::vector<double>> &costMatrix);
       
        void auctionAssignement(
            std::vector<std::vector<double>> &costMatrix,
            std::vector<ttk::MatchingType> &matching);
       


        void performMatchings(
          std::vector<DiagramType> persistenceDiagrams, 
          std::vector<std::vector<MatchingType>> &maximaMatchings,
          std::vector<std::vector<MatchingType>> &sad_1_Matchings,
          std::vector<std::vector<MatchingType>> &sad_2_Matchings,
          std::vector<std::vector<MatchingType>> &minimaMatchings, 
          int fieldNumber);

    
      };
}