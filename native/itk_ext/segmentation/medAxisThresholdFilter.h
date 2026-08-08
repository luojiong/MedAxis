/*=========================================================================

  medAxisThresholdFilter.h

  Global thresholding with 9 automatic methods plus manual range mode.
  Output is a binary label image (InsideValue / OutsideValue).

=========================================================================*/
#ifndef medAxisThresholdFilter_h
#define medAxisThresholdFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

namespace medaxis
{

enum class ThresholdMethod
{
    Otsu,
    Adaptive,
    IsoData,
    MaxEntropy,
    Moments,
    RenyiEntropy,
    Huang,
    Intermodes,
    Li,
    Manual
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT ThresholdFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(ThresholdFilter);

    using Self = ThresholdFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(ThresholdFilter);

    using OutputPixelType = typename TOutputImage::PixelType;

    itkSetMacro(Method, ThresholdMethod);
    itkGetConstReferenceMacro(Method, ThresholdMethod);

    /// Manual mode range.
    itkSetMacro(LowerThreshold, double);
    itkGetConstReferenceMacro(LowerThreshold, double);
    itkSetMacro(UpperThreshold, double);
    itkGetConstReferenceMacro(UpperThreshold, double);

    itkSetMacro(InsideValue, OutputPixelType);
    itkGetConstReferenceMacro(InsideValue, OutputPixelType);
    itkSetMacro(OutsideValue, OutputPixelType);
    itkGetConstReferenceMacro(OutsideValue, OutputPixelType);

    itkSetMacro(NumberOfHistogramBins, unsigned int);
    itkGetConstReferenceMacro(NumberOfHistogramBins, unsigned int);

    /// Adaptive-threshold local neighborhood radius (voxels).
    itkSetMacro(AdaptiveRadius, unsigned int);
    itkGetConstReferenceMacro(AdaptiveRadius, unsigned int);

    /// The computed threshold (valid after Update()).
    itkGetConstReferenceMacro(ComputedThreshold, double);

protected:
    ThresholdFilter();
    ~ThresholdFilter() override = default;

    void GenerateData() override;

private:
    template <typename TFilter>
    void RunHistogramFilter();

    ThresholdMethod   m_Method = ThresholdMethod::Otsu;
    double            m_LowerThreshold = 0.0;
    double            m_UpperThreshold = 0.0;
    OutputPixelType   m_InsideValue = static_cast<OutputPixelType>(1);
    OutputPixelType   m_OutsideValue = static_cast<OutputPixelType>(0);
    unsigned int      m_NumberOfHistogramBins = 128;
    unsigned int      m_AdaptiveRadius = 5;
    double            m_ComputedThreshold = 0.0;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisThresholdFilter.hxx"
#endif

#endif // medAxisThresholdFilter_h
