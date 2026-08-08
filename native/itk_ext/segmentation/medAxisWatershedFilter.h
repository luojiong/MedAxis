/*=========================================================================

  medAxisWatershedFilter.h

  Watershed segmentation with two modes:
    Gradient      - classic watershed on a gradient-magnitude image
                    (itk::WatershedImageFilter)
    Morphological - Meyer's morphological watershed
                    (itk::MorphologicalWatershedImageFilter)

=========================================================================*/
#ifndef medAxisWatershedFilter_h
#define medAxisWatershedFilter_h

#include "itkImageToImageFilter.h"

namespace medaxis
{

enum class WatershedMode
{
    Gradient,
    Morphological
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT WatershedFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(WatershedFilter);

    using Self = WatershedFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(WatershedFilter);

    itkSetMacro(Mode, WatershedMode);
    itkGetConstReferenceMacro(Mode, WatershedMode);

    /// Water level (fraction of the max gradient / morphological level).
    itkSetMacro(Level, double);
    itkGetConstReferenceMacro(Level, double);

    /// Gradient threshold below which basins are merged (0..1).
    itkSetMacro(Threshold, double);
    itkGetConstReferenceMacro(Threshold, double);

    /// Sigma of the gradient-magnitude pre-smoothing (world units).
    itkSetMacro(GradientSigma, double);
    itkGetConstReferenceMacro(GradientSigma, double);

    itkSetMacro(MarkWatershedLine, bool);
    itkGetConstReferenceMacro(MarkWatershedLine, bool);
    itkBooleanMacro(MarkWatershedLine);

protected:
    WatershedFilter();
    ~WatershedFilter() override = default;

    void GenerateData() override;

private:
    WatershedMode m_Mode = WatershedMode::Morphological;
    double        m_Level = 0.5;
    double        m_Threshold = 0.01;
    double        m_GradientSigma = 1.0;
    bool          m_MarkWatershedLine = true;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisWatershedFilter.hxx"
#endif

#endif // medAxisWatershedFilter_h
