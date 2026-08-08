/*=========================================================================

  medAxisBSplineRegistration.h

  Deformable BSpline registration (ITKv4) with configurable control-point
  mesh size and a Mattes Mutual Information metric.
  Input 0 = fixed, input 1 = moving.
  Output = moving image resampled into the fixed image grid.

=========================================================================*/
#ifndef medAxisBSplineRegistration_h
#define medAxisBSplineRegistration_h

#include "itkProcessObject.h"
#include "itkBSplineTransform.h"

namespace medaxis
{

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT BSplineRegistration : public itk::ProcessObject
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(BSplineRegistration);

    using Self = BSplineRegistration;
    using Superclass = itk::ProcessObject;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int Dimension = TFixedImage::ImageDimension;
    static constexpr unsigned int SplineOrder = 3;

    using TransformType = itk::BSplineTransform<double, Dimension, SplineOrder>;
    using OutputImageType = TMovingImage;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(BSplineRegistration);

    void SetFixedImage(const TFixedImage* image);
    const TFixedImage* GetFixedImage() const;

    void SetMovingImage(const TMovingImage* image);
    const TMovingImage* GetMovingImage() const;

    /// Number of control-point mesh subdivisions per axis.
    itkSetMacro(MeshSize, unsigned int);
    itkGetConstReferenceMacro(MeshSize, unsigned int);

    itkSetMacro(NumberOfIterations, unsigned int);
    itkGetConstReferenceMacro(NumberOfIterations, unsigned int);

    itkSetMacro(LearningRate, double);
    itkGetConstReferenceMacro(LearningRate, double);

    itkSetMacro(MetricSamplingPercentage, double);
    itkGetConstReferenceMacro(MetricSamplingPercentage, double);

    itkSetMacro(NumberOfHistogramBins, unsigned int);
    itkGetConstReferenceMacro(NumberOfHistogramBins, unsigned int);

    const TransformType* GetTransform() const { return m_Transform.GetPointer(); }

    itkGetConstReferenceMacro(FinalMetricValue, double);

    OutputImageType* GetOutput();

protected:
    BSplineRegistration();
    ~BSplineRegistration() override = default;

    void GenerateData() override;

    using Superclass::MakeOutput;
    DataObjectPointer MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
    typename TransformType::Pointer m_Transform;
    unsigned int                    m_MeshSize = 8;
    unsigned int                    m_NumberOfIterations = 200;
    double                          m_LearningRate = 0.01;
    double                          m_MetricSamplingPercentage = 0.2;
    unsigned int                    m_NumberOfHistogramBins = 32;
    double                          m_FinalMetricValue = 0.0;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisBSplineRegistration.hxx"
#endif

#endif // medAxisBSplineRegistration_h
