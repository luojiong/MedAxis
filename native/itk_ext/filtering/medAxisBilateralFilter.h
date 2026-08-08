/*=========================================================================

  medAxisBilateralFilter.h

  Edge-preserving smoothing. Thin wrapper around itk::BilateralImageFilter
  with a MedAxis-consistent parameter surface.

=========================================================================*/
#ifndef medAxisBilateralFilter_h
#define medAxisBilateralFilter_h

#include "itkImageToImageFilter.h"
#include "itkBilateralImageFilter.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT BilateralFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(BilateralFilter);

    using Self = BilateralFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(BilateralFilter);

    static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

    using InputImageType = TInputImage;
    using OutputImageType = TOutputImage;
    using InputPixelType = typename TInputImage::PixelType;
    using OutputPixelType = typename TOutputImage::PixelType;

    /// Spatial extent of the filter (world units).
    itkSetMacro(DomainSigma, double);
    itkGetConstReferenceMacro(DomainSigma, double);

    /// Radiometric extent of the filter (intensity units).
    itkSetMacro(RangeSigma, double);
    itkGetConstReferenceMacro(RangeSigma, double);

    /// Number of samples of the range Gaussian (accuracy vs speed).
    itkSetMacro(NumberOfRangeGaussianSamples, unsigned int);
    itkGetConstReferenceMacro(NumberOfRangeGaussianSamples, unsigned int);

protected:
    BilateralFilter();
    ~BilateralFilter() override = default;

    void GenerateData() override;

private:
    double       m_DomainSigma = 4.0;
    double       m_RangeSigma = 50.0;
    unsigned int m_NumberOfRangeGaussianSamples = 25;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisBilateralFilter.hxx"
#endif

#endif // medAxisBilateralFilter_h
