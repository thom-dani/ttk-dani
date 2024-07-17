/// \ingroup base
/// \class ttk::TopologicalSimplification
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \author Guillaume Favelier <guillaume.favelier@lip6.fr>
/// \date February 2016
///
/// \brief TTK processing package for the topological simplification of scalar
/// data.
///
/// Given an input scalar field and a list of critical points to remove, this
/// class minimally edits the scalar field such that the listed critical points
/// disappear. This procedure is useful to speedup subsequent topological data
/// analysis when outlier critical points can be easily identified. It is
/// also useful for data simplification.
///
/// \b Related \b publications \n
/// "Generalized Topological Simplification of Scalar Fields on Surfaces" \n
/// Julien Tierny, Valerio Pascucci \n
/// Proc. of IEEE VIS 2012.\n
/// IEEE Transactions on Visualization and Computer Graphics, 2012.
///
/// "Localized Topological Simplification of Scalar Data"
/// Jonas Lukasczyk, Christoph Garth, Ross Maciejewski, Julien Tierny
/// Proc. of IEEE VIS 2020.
/// IEEE Transactions on Visualization and Computer Graphics
///
/// \sa ttkTopologicalSimplification.cpp %for a usage example.
///
/// \b Online \b examples: \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/1manifoldLearning/">1-Manifold
///   Learning example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/1manifoldLearningCircles/">1-Manifold
///   Learning Circles example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/2manifoldLearning/">
///   2-Manifold Learning example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/BuiltInExample1/">BuiltInExample1
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/contourTreeAlignment/">Contour
///   Tree Alignment example</a> \n
///   - <a href="https://topology-tool-kit.github.io/examples/ctBones/">CT Bones
///   example</a> \n
///   - <a href="https://topology-tool-kit.github.io/examples/dragon/">Dragon
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/harmonicSkeleton/">
///   Harmonic Skeleton example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/imageProcessing/">Image
///   Processing example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/interactionSites/">
///   Interaction sites</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/karhunenLoveDigits64Dimensions/">Karhunen-Love
///   Digits 64-Dimensions example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/morsePersistence/">Morse
///   Persistence example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/morseSmaleQuadrangulation/">Morse-Smale
///   Quadrangulation example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 0 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 1 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 2 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 3 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/persistenceClustering0/">Persistence
///   clustering 4 example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/tectonicPuzzle/">Tectonic
///   Puzzle example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/tribute/">Tribute
///   example</a> \n
///   - <a
///   href="https://topology-tool-kit.github.io/examples/uncertainStartingVortex/">
///   Uncertain Starting Vortex example</a> \n

#pragma once

// base code includes
#include <BackendTopologicalOptimization.h>
#include <Debug.h>
#include <LegacyTopologicalSimplification.h>
#include <LocalizedTopologicalSimplification.h>
#include <PersistenceDiagram.h>
#include <Triangulation.h>

#include <cmath>
#include <set>
#include <tuple>
#include <type_traits>

namespace ttk {

  class TopologicalSimplification : virtual public Debug {
  public:
    TopologicalSimplification();

    enum class BACKEND { LEGACY, LTS, PS };
    /*
     * Either execute this file "legacy" algorithm, or the
     * lts algorithm. The choice depends on the value of the variable backend_.
     * Default is lts (localized).
     */
    template <typename dataType, typename triangulationType>
    int execute(const dataType *const inputScalars,
                dataType *const outputScalars,
                const SimplexId *const identifiers,
                const SimplexId *const inputOffsets,
                SimplexId *const offsets,
                const SimplexId constraintNumber,
                const bool addPerturbation,
                triangulationType &triangulation,
                const ttk::DiagramType &constraintDiagram = {});

    inline void setBackend(const BACKEND arg) {
      backend_ = arg;
    }

    inline int preconditionTriangulation(AbstractTriangulation *triangulation) {
      switch(backend_) {
        case BACKEND::LEGACY:
          legacyObject_.setDebugLevel(debugLevel_);
          legacyObject_.setThreadNumber(threadNumber_);
          legacyObject_.preconditionTriangulation(triangulation);
          break;

        case BACKEND::LTS:
          ltsObject_.setDebugLevel(debugLevel_);
          ltsObject_.setThreadNumber(threadNumber_);
          ltsObject_.preconditionTriangulation(triangulation);
          break;

        case BACKEND::PS:
          PSObject_.setDebugLevel(debugLevel_);
          PSObject_.setThreadNumber(threadNumber_);
          PSObject_.preconditionTriangulation(triangulation);
          break;

        default:
          this->printErr(
            "Error, the backend for topological simplification is invalid");
          return -1;
      }
      return 0;
    }

  protected:
    BACKEND backend_{BACKEND::LTS};
    LegacyTopologicalSimplification legacyObject_;
    lts::LocalizedTopologicalSimplification ltsObject_;
    ttk::BackendTopologicalOptimization PSObject_;

    SimplexId vertexNumber_{};
    bool UseFastPersistenceUpdate{true};
    bool FastAssignmentUpdate{true};
    int EpochNumber{1000};

    // if PDCMethod == 0 then we use Progressive approach
    // if PDCMethod == 1 then we use Classical Auction approach
    int PDCMethod{1};

    // if MethodOptimization == 0 then we use direct optimization
    // if MethodOptimization == 1 then we use Adam
    int MethodOptimization{0};

    // if FinePairManagement == 0 then we let the algorithm choose
    // if FinePairManagement == 1 then we fill the domain
    // if FinePairManagement == 2 then we cut the domain
    int FinePairManagement{0};

    // Adam
    bool ChooseLearningRate{false};
    double LearningRate{0.0001};

    // Direct Optimization : Gradient Step Size
    double Alpha{0.5};

    // Stopping criterion: when the loss becomes less than a percentage (e.g.
    // 1%) of the original loss (between input diagram and simplified diagram)
    double CoefStopCondition{0.01};

    //
    bool OptimizationWithoutMatching{false};
    int ThresholdMethod{1};
    double Threshold{0.01};
    int LowerThreshold{-1};
    int UpperThreshold{2};
    int PairTypeToDelete{1};

    bool ConstraintAveraging{true};
  };
} // namespace ttk

template <typename dataType, typename triangulationType>
int ttk::TopologicalSimplification::execute(
  const dataType *const inputScalars,
  dataType *const outputScalars,
  const SimplexId *const identifiers,
  const SimplexId *const inputOffsets,
  SimplexId *const offsets,
  const SimplexId constraintNumber,
  const bool addPerturbation,
  triangulationType &triangulation,
  const ttk::DiagramType &constraintDiagram) {
  switch(backend_) {
    case BACKEND::LTS:
      return ltsObject_
        .removeUnauthorizedExtrema<dataType, SimplexId, triangulationType>(
          outputScalars, offsets, &triangulation, identifiers, constraintNumber,
          addPerturbation);
    case BACKEND::LEGACY:
      return legacyObject_.execute(inputScalars, outputScalars, identifiers,
                                   inputOffsets, offsets, constraintNumber,
                                   triangulation);

    case BACKEND::PS:
      PSObject_.setUseFastPersistenceUpdate(UseFastPersistenceUpdate);
      PSObject_.setFastAssignmentUpdate(FastAssignmentUpdate);
      PSObject_.setEpochNumber(EpochNumber);
      PSObject_.setPDCMethod(PDCMethod);
      PSObject_.setMethodOptimization(MethodOptimization);
      PSObject_.setFinePairManagement(FinePairManagement);
      PSObject_.setChooseLearningRate(ChooseLearningRate);
      PSObject_.setLearningRate(LearningRate);
      PSObject_.setAlpha(Alpha);
      PSObject_.setCoefStopCondition(CoefStopCondition);
      PSObject_.setOptimizationWithoutMatching(OptimizationWithoutMatching);
      PSObject_.setThresholdMethod(ThresholdMethod);
      PSObject_.setThresholdPersistence(Threshold);
      PSObject_.setLowerThreshold(LowerThreshold);
      PSObject_.setUpperThreshold(UpperThreshold);
      PSObject_.setPairTypeToDelete(PairTypeToDelete);
      PSObject_.setConstraintAveraging(ConstraintAveraging);

      return PSObject_.execute(inputScalars, outputScalars, offsets,
                               &triangulation, constraintDiagram);
    default:
      this->printErr(
        "Error, the backend for topological simplification is invalid");
      return -1;
  }
}
