/*=========================================================================

  medAxisRigidRegistration.hxx

=========================================================================*/
#ifndef medAxisRigidRegistration_hxx
#define medAxisRigidRegistration_hxx

#include "medAxisRigidRegistration.h"

#include "itkImageRegistrationMethodv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkResampleImageFilter.h"
#include "itkCenteredTransformInitializer.h"

namespace medaxis
{

template <typename TFixedImage, typename TMovingImage>
RigidRegistration<TFixedImage, TMovingImage>::RigidRegistration()
{
    this->SetNumberOfRequiredInputs(2);
    this->SetNumberOfRequiredOutputs(1);
    this->SetNthOutput(0, this->MakeOutput(0));
    m_Transform = TransformType::New();
}

template <typename TFixedImage, typename TMovingImage>
typename itk::ProcessObject::DataObjectPointer
RigidRegistration<TFixedImage, TMovingImage>::MakeOutput(DataObjectPointerArraySizeType)
{
    return OutputImageType::New().GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
void
RigidRegistration<TFixedImage, TMovingImage>::SetFixedImage(const TFixedImage* image)
{
    this->ProcessObject::SetInput(0, const_cast<TFixedImage*>(image));
}

template <typename TFixedImage, typename TMovingImage>
const TFixedImage*
RigidRegistration<TFixedImage, TMovingImage>::GetFixedImage() const
{
    return static_cast<const TFixedImage*>(this->ProcessObject::GetInput(0));
}

template <typename TFixedImage, typename TMovingImage>
void
RigidRegistration<TFixedImage, TMovingImage>::SetMovingImage(const TMovingImage* image)
{
    this->ProcessObject::SetInput(1, const_cast<TMovingImage*>(image));
}

template <typename TFixedImage, typename TMovingImage>
const TMovingImage*
RigidRegistration<TFixedImage, TMovingImage>::GetMovingImage() const
{
    return static_cast<const TMovingImage*>(this->ProcessObject::GetInput(1));
}

template <typename TFixedImage, typename TMovingImage>
typename RigidRegistration<TFixedImage, TMovingImage>::OutputImageType*
RigidRegistration<TFixedImage, TMovingImage>::GetOutput()
{
    return static_cast<OutputImageType*>(this->ProcessObject::GetOutput(0));
}

template <typename TFixedImage, typename TMovingImage>
void
RigidRegistration<TFixedImage, TMovingImage>::GenerateData()
{
    const auto* fixed = this->GetFixedImage();
    const auto* moving = this->GetMovingImage();

    using RegistrationType =
        itk::ImageRegistrationMethodv4<TFixedImage, TMovingImage, TransformType>;
    typename RegistrationType::Pointer registration = RegistrationType::New();

    registration->SetFixedImage(const_cast<TFixedImage*>(fixed));
    registration->SetMovingImage(const_cast<TMovingImage*>(moving));

    // Initialize transform centers on image geometry.
    m_Transform = TransformType::New();
    using InitializerType =
        itk::CenteredTransformInitializer<TransformType, TFixedImage, TMovingImage>;
    typename InitializerType::Pointer initializer = InitializerType::New();
    initializer->SetTransform(m_Transform);
    initializer->SetFixedImage(const_cast<TFixedImage*>(fixed));
    initializer->SetMovingImage(const_cast<TMovingImage*>(moving));
    initializer->SetOperationMode(InitializerType::MOMENTS);
    initializer->InitializeTransform();

    registration->SetInitialTransform(m_Transform.GetPointer());

    using MetricType = itk::MattesMutualInformationImageToImageMetricv4<TFixedImage, TMovingImage>;
    typename MetricType::Pointer metric = MetricType::New();
    metric->SetNumberOfHistogramBins(m_NumberOfHistogramBins);
    metric->SetUseMovingImageGradientFilter(false);
    metric->SetUseFixedImageGradientFilter(false);
    registration->SetMetric(metric);

    registration->SetMetricSamplingStrategy(RegistrationType::REGULAR);
    registration->SetMetricSamplingPercentage(m_MetricSamplingPercentage);

    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    typename OptimizerType::Pointer optimizer = OptimizerType::New();
    optimizer->SetNumberOfIterations(m_NumberOfIterations);
    optimizer->SetLearningRate(1.0);
    optimizer->SetMinimumStepLength(0.001);
    optimizer->SetRelaxationFactor(0.5);
    optimizer->SetReturnBestParametersAndValue(true);
    registration->SetOptimizer(optimizer);

    registration->SetNumberOfLevels(1);
    registration->Update();

    m_FinalMetricValue = optimizer->GetValue();

    // Resample moving image into the fixed grid using the final transform.
    using ResampleType = itk::ResampleImageFilter<TMovingImage, OutputImageType>;
    typename ResampleType::Pointer resampler = ResampleType::New();
    resampler->SetInput(const_cast<TMovingImage*>(moving));
    resampler->SetTransform(m_Transform.GetPointer());
    resampler->SetSize(fixed->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(fixed->GetOrigin());
    resampler->SetOutputSpacing(fixed->GetSpacing());
    resampler->SetOutputDirection(fixed->GetDirection());
    resampler->SetDefaultPixelValue(static_cast<typename OutputImageType::PixelType>(0));
    resampler->Update();

    this->GraftOutput(resampler->GetOutput());
}

} // namespace medaxis

#endif // medAxisRigidRegistration_hxx
