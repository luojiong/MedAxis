/*=========================================================================

  medAxisAffineRegistration.h

  Affine registration (12 DOF) using the ITKv4 framework with a Mattes
  Mutual Information metric. Input 0 = fixed, input 1 = moving.
  Output = moving image resampled into the fixed image grid.

=========================================================================*/
#ifndef medAxisAffineRegistration_h
#define medAxisAffineRegistration_h

#include "itkProcessObject.h"
#include "itkAffineTransform.h"

namespace medaxis
{

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT AffineRegistration : public itk::ProcessObject
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(AffineRegistration);

    using Self = AffineRegistration;
    using Superclass = itk::ProcessObject;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int Dimension = TFixedImage::ImageDimension;

    using TransformType = itk::AffineTransform<double, Dimension>;
    using OutputImageType = TMovingImage;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(AffineRegistration);

    void SetFixedImage(const TFixedImage* image);
    const TFixedImage* GetFixedImage() const;

    void SetMovingImage(const TMovingImage* image);
    const TMovingImage* GetMovingImage() const;

    itkSetMacro(NumberOfIterations, unsigned int);
    itkGetConstReferenceMacro(NumberOfIterations, unsigned int);

    itkSetMacro(MetricSamplingPercentage, double);
    itkGetConstReferenceMacro(MetricSamplingPercentage, double);

    itkSetMacro(NumberOfHistogramBins, unsigned int);
    itkGetConstReferenceMacro(NumberOfHistogramBins, unsigned int);

    const TransformType* GetTransform() const { return m_Transform.GetPointer(); }

    itkGetConstReferenceMacro(FinalMetricValue, double);

    OutputImageType* GetOutput();

protected:
    AffineRegistration();
    ~AffineRegistration() override = default;

    void GenerateData() override;

    using Superclass::MakeOutput;
    DataObjectPointer MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
    typename TransformType::Pointer m_Transform;
    unsigned int                    m_NumberOfIterations = 300;
    double                          m_MetricSamplingPercentage = 0.2;
    unsigned int                    m_NumberOfHistogramBins = 32;
    double                          m_FinalMetricValue = 0.0;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisAffineRegistration.hxx"
#endif

#endif // medAxisAffineRegistration_h
