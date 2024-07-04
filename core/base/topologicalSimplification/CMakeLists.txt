ttk_add_base_library(topologicalSimplification
  SOURCES
    TopologicalSimplification.cpp
  HEADERS
    TopologicalSimplification.h
    BackendTopologicalOptimization.h 
  DEPENDS
    triangulation
    legacyTopologicalSimplification
    localizedTopologicalSimplification
    persistenceDiagram
    persistenceDiagramClustering
    )

if(TTK_ENABLE_TORCH)
  target_include_directories(topologicalSimplification PUBLIC ${TORCH_INCLUDE_DIRS})
  target_compile_options(topologicalSimplification PUBLIC "${TORCH_CXX_FLAGS}")
  target_link_libraries(topologicalSimplification PUBLIC "${TORCH_LIBRARIES}")
  target_compile_definitions(topologicalSimplification PUBLIC TTK_ENABLE_TORCH)
endif()