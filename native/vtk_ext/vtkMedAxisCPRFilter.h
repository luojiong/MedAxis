/*=========================================================================

  vtkMedAxisCPRFilter.h

  Curved Planar Reformation (CPR) filter for MedAxis.

  Given a space curve (e.g. a vessel centerline) and per-point normals,
  produces a straightened 2D image: X axis = arc length along the curve,
  Y axis = offset along the normal, Z = 1.

=========================================================================*/
#ifndef vtkMedAxisCPRFilter_h
#define vtkMedAxisCPRFilter_h

#include "vtkImageAlgorithm.h"

class vtkPoints;
class vtkFloatArray;

class vtkMedAxisCPRFilter : public vtkImageAlgorithm
{
public:
    static vtkMedAxisCPRFilter* New();
    vtkTypeMacro(vtkMedAxisCPRFilter, vtkImageAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /// Centerline / curve sample points (world coordinates).
    virtual void SetCurvePoints(vtkPoints*);
    vtkGetObjectMacro(CurvePoints, vtkPoints);

    /// Per-curve-point normals, 3 floats per point (nx, ny, nz), defining
    /// the in-plane Y axis of the straightened image. Must contain exactly
    /// 3 * CurvePoints->GetNumberOfPoints() tuples. If not set, a heuristic
    /// parallel-transport normal is computed.
    virtual void SetCurveNormals(vtkFloatArray*);
    vtkGetObjectMacro(CurveNormals, vtkFloatArray);

    /// Half-width of the straightened image in world units.
    vtkSetClampMacro(Thickness, double, 1.0e-6, VTK_DOUBLE_MAX);
    vtkGetMacro(Thickness, double);

    /// Number of samples across the image Y direction.
    vtkSetClampMacro(NumRows, int, 2, VTK_INT_MAX);
    vtkGetMacro(NumRows, int);

protected:
    vtkMedAxisCPRFilter();
    ~vtkMedAxisCPRFilter() override;

    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
    int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

    vtkPoints*     CurvePoints = nullptr;
    vtkFloatArray* CurveNormals = nullptr;
    double         Thickness = 10.0;
    int            NumRows = 128;

private:
    vtkMedAxisCPRFilter(const vtkMedAxisCPRFilter&) = delete;
    void operator=(const vtkMedAxisCPRFilter&) = delete;
};

#endif // vtkMedAxisCPRFilter_h
