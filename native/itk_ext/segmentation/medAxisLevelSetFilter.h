/*=========================================================================

  medAxisLevelSetFilter.h

  Level-set segmentation with four ITK backends:
    Geodesic     - GeodesicActiveContourLevelSetImageFilter
    ShapeDetection - ShapeDetectionLevelSetImageFilter
    Threshold    - ThresholdSegmentationLevelSetImageFilter
    Laplacian    - LaplacianSegmentationLevelSetImageFilter

  The initial level set is a signed distance map of a ball seeded at
  SeedPoint with SeedRadius.

=========================================================================*/
#ifndef medAxisLevelSetFilter_h
#define medAxisLevelSetFilter_h

#include "itkImageToImageFilter.h"
#include "itkPoint.h"

namespace medaxis
{

enum class LevelSetMode
{
    Geodesic,
    ShapeDetection,
    Threshold,
    Laplacian
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT LevelSetFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(LevelSetFilter);

    using Self = LevelSetFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;
    using PointType = itk::Point<double, ImageDimension>;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(LevelSetFilter);

    itkSetMacro(Mode, LevelSetMode);
    itkGetConstReferenceMacro(Mode, LevelSetMode);

    /// Seed (world coordinates) and radius of the initial contour ball.
    itkSetMacro(SeedPoint, PointType);
    itkGetConstReferenceMacro(SeedPoint, PointType);
    itkSetMacro(SeedRadius, double);
    itkGetConstReferenceMacro(SeedRadius, double);

    itkSetMacro(MaxIterations, unsigned int);
    itkGetConstReferenceMacro(MaxIterations, unsigned int);

    itkSetMacro(PropagationScaling, double);
    itkGetConstReferenceMacro(PropagationScaling, double);
    itkSetMacro(CurvatureScaling, double);
    itkGetConstReferenceMacro(CurvatureScaling, double);
    itkSetMacro(AdvectionScaling, double);
    itkGetConstReferenceMacro(AdvectionScaling, double);

    /// Threshold mode window.
    itkSetMacro(LowerThreshold, double);
    itkGetConstReferenceMacro(LowerThreshold, double);
    itkSetMacro(UpperThreshold, double);
    itkGetConstReferenceMacro(UpperThreshold, double);

    itkSetMacro(OutputValue, OutputPixelType);
    itkGetConstReferenceMacro(OutputValue, OutputPixelType);

protected:
    LevelSetFilter();
    ~LevelSetFilter() override = default;

    void GenerateData() override;

private:
    LevelSetMode   m_Mode = LevelSetMode::Geodesic;
    PointType      m_SeedPoint;
    double         m_SeedRadius = 5.0;
    unsigned int   m_MaxIterations = 300;
    double         m_PropagationScaling = 1.0;
    double         m_CurvatureScaling = 1.0;
    double         m_AdvectionScaling = 1.0;
    double         m_LowerThreshold = 0.0;
    double         m_UpperThreshold = 0.0;
    OutputPixelType m_OutputValue = static_cast<OutputPixelType>(1);
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisLevelSetFilter.hxx"
#endif

#endif // medAxisLevelSetFilter_h
