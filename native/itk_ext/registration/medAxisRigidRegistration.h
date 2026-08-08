/*=========================================================================

  medAxisRigidRegistration.h

  Rigid registration (ITKv4 framework) with a VersorRigid3D transform and
  Mattes Mutual Information metric. Input 0 = fixed, input 1 = moving.
  Output = moving image resampled into the fixed image grid.

=========================================================================*/
#ifndef medAxisRigidRegistration_h
#define medAxisRigidRegistration_h

#include "itkProcessObject.h"
#include "itkVersorRigid3DTransform.h"

namespace medaxis
{

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT RigidRegistration : public itk::ProcessObject
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(RigidRegistration);

    using Self = RigidRegistration;
    using Superclass = itk::ProcessObject;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int Dimension = TFixedImage::ImageDimension;

    using TransformType = itk::VersorRigid3DTransform<double>;
    using OutputImageType = TMovingImage;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(RigidRegistration);

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

    /// Resulting rigid transform (fixed -> moving space). Valid after Update().
    const TransformType* GetTransform() const { return m_Transform.GetPointer(); }

    /// Final metric value after optimization.
    itkGetConstReferenceMacro(FinalMetricValue, double);

    OutputImageType* GetOutput();

protected:
    RigidRegistration();
    ~RigidRegistration() override = default;

    void GenerateData() override;

    using Superclass::MakeOutput;
    DataObjectPointer MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
    typename TransformType::Pointer m_Transform;
    unsigned int                    m_NumberOfIterations = 200;
    double                          m_MetricSamplingPercentage = 0.2;
    unsigned int                    m_NumberOfHistogramBins = 32;
    double                          m_FinalMetricValue = 0.0;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisRigidRegistration.hxx"
#endif

#endif // medAxisRigidRegistration_h
