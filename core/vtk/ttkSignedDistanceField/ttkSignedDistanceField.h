#pragma once

// VTK Module
#include <ttkSignedDistanceFieldModule.h>

// ttk code includes
#include <SignedDistanceField.h>
#include <ttkAlgorithm.h>

class vtkImageData;

class TTKSIGNEDDISTANCEFIELD_EXPORT ttkSignedDistanceField
  : public ttkAlgorithm,
    protected ttk::SignedDistanceField {
public:
  static ttkSignedDistanceField *New();
  vtkTypeMacro(ttkSignedDistanceField, ttkAlgorithm);

  ///@{
  /**
   * Set/Get sampling dimension along each axis. Default will be [10,10,10]
   */
  vtkSetVector3Macro(SamplingDimensions, int);
  vtkGetVector3Macro(SamplingDimensions, int);
  ///@}

  vtkSetMacro(ExpandBox, bool);
  vtkGetMacro(ExpandBox, bool);

  vtkSetMacro(Backend, int);
  vtkGetMacro(Backend, int);

  vtkSetMacro(FastMarchingOrder, int);
  vtkGetMacro(FastMarchingOrder, int);

  /**
   * Get the output data for this algorithm.
   */
  vtkImageData *GetOutput();

protected:
  ttkSignedDistanceField();

  // Usual data generation method
  vtkTypeBool ProcessRequest(vtkInformation *,
                             vtkInformationVector **,
                             vtkInformationVector *) override;
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;
  virtual int RequestInformation(vtkInformation *,
                                 vtkInformationVector **,
                                 vtkInformationVector *);
  virtual int RequestUpdateExtent(vtkInformation *,
                                  vtkInformationVector **,
                                  vtkInformationVector *);
  int FillInputPortInformation(int, vtkInformation *) override;
  int FillOutputPortInformation(int, vtkInformation *) override;

  void computeOutputInformation(vtkInformationVector **inputVector);

  int SamplingDimensions[3];
  bool ExpandBox = true;
  int Backend = 0;
  int FastMarchingOrder = 1;

private:
  std::array<int, 6> DataExtent{0, 0, 0, 0, 0, 0};
  std::array<double, 3> Origin{0.0, 0.0, 0.0};
};
