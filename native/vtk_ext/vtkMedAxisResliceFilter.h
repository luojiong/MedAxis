/*=========================================================================

  vtkMedAxisResliceFilter.h

  Multi-planar reslice filter for MedAxis. Wraps vtkImageReslice with
  MedAxis-specific conveniences (slab thickness in slice units, explicit
  interpolation mode enum).

=========================================================================*/
#ifndef vtkMedAxisResliceFilter_h
#define vtkMedAxisResliceFilter_h

#include "vtkImageAlgorithm.h"

class vtkMatrix4x4;

class vtkMedAxisResliceFilter : public vtkImageAlgorithm
{
public:
    static vtkMedAxisResliceFilter* New();
    vtkTypeMacro(vtkMedAxisResliceFilter, vtkImageAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /// Interpolation modes (mirrors vtkImageReslice enums).
    enum InterpolationMode
    {
        NEAREST_NEIGHBOR = 0,
        TRILINEAR        = 1,
        BICUBIC          = 2,
        SINC             = 3
    };

    /// The 4x4 reslice axes matrix. Columns 0-2 are the output x/y/z axes
    /// expressed in input coordinates; column 3 is the output origin.
    virtual void SetResliceAxes(vtkMatrix4x4*);
    vtkGetObjectMacro(ResliceAxes, vtkMatrix4x4);

    vtkSetClampMacro(InterpolationMode, int, NEAREST_NEIGHBOR, SINC);
    vtkGetMacro(InterpolationMode, int);
    void SetInterpolationModeToNearestNeighbor() { this->SetInterpolationMode(NEAREST_NEIGHBOR); }
    void SetInterpolationModeToTrilinear()       { this->SetInterpolationMode(TRILINEAR); }
    void SetInterpolationModeToBicubic()         { this->SetInterpolationMode(BICUBIC); }
    void SetInterpolationModeToSinc()            { this->SetInterpolationMode(SINC); }

    /// Slab thickness expressed in number of output slices (MPR slab / MIP
    /// style averaging). 1 == single slice.
    vtkSetClampMacro(SlabThickness, int, 1, VTK_INT_MAX);
    vtkGetMacro(SlabThickness, int);

    /// Output slice spacing along the reslice Z axis (world units).
    vtkSetMacro(OutputSpacing, double);
    vtkGetMacro(OutputSpacing, double);

protected:
    vtkMedAxisResliceFilter();
    ~vtkMedAxisResliceFilter() override;

    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

    vtkMatrix4x4* ResliceAxes = nullptr;
    int           InterpolationMode = TRILINEAR;
    int           SlabThickness = 1;
    double        OutputSpacing = 1.0;

private:
    vtkMedAxisResliceFilter(const vtkMedAxisResliceFilter&) = delete;
    void operator=(const vtkMedAxisResliceFilter&) = delete;
};

#endif // vtkMedAxisResliceFilter_h
