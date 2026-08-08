/*=========================================================================

  medAxisAutoFillFilter.h

  Inter-slice label interpolation: fills the gaps between sparsely
  labeled slices of a 3D label volume.

  Modes:
    Linear       - per-voxel linear blend of the bounding slice masks
    Morphological- skeleton falls back to linear (Phase 1: true
                   morphological interpolation via dilation/erosion)
    BSpline      - skeleton falls back to linear (Phase 1: B-spline
                   contour fitting)

=========================================================================*/
#ifndef medAxisAutoFillFilter_h
#define medAxisAutoFillFilter_h

#include "itkImageToImageFilter.h"

namespace medaxis
{

enum class AutoFillMode
{
    Linear,
    Morphological,
    BSpline
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT AutoFillFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(AutoFillFilter);

    using Self = AutoFillFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(AutoFillFilter);

    itkSetMacro(Mode, AutoFillMode);
    itkGetConstReferenceMacro(Mode, AutoFillMode);

    /// Slice axis to interpolate along (0=x, 1=y, 2=z).
    itkSetClampMacro(SliceAxis, unsigned int, 0, TInputImage::ImageDimension - 1);
    itkGetConstReferenceMacro(SliceAxis, unsigned int);

    /// Label value considered "annotated".
    itkSetMacro(LabelValue, OutputPixelType);
    itkGetConstReferenceMacro(LabelValue, OutputPixelType);

protected:
    AutoFillFilter();
    ~AutoFillFilter() override = default;

    void GenerateData() override;

private:
    AutoFillMode  m_Mode = AutoFillMode::Linear;
    unsigned int  m_SliceAxis = 2;
    OutputPixelType m_LabelValue = static_cast<OutputPixelType>(1);
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisAutoFillFilter.hxx"
#endif

#endif // medAxisAutoFillFilter_h
