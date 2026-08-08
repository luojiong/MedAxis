/*=========================================================================

  medAxisDemonsRegistration.hxx

=========================================================================*/
#ifndef medAxisDemonsRegistration_hxx
#define medAxisDemonsRegistration_hxx

#include "medAxisDemonsRegistration.h"

#include "itkDemonsRegistrationFilter.h"
#include "itkDiffeomorphicDemonsRegistrationFilter.h"
#include "itkFastSymmetricForcesDemonsRegistrationFilter.h"
#include "itkCastImageFilter.h"

namespace medaxis
{

template <typename TFixedImage, typename TMovingImage>
DemonsRegistration<TFixedImage, TMovingImage>::DemonsRegistration()
{
    this->SetNumberOfRequiredInputs(2);
    this->SetNumberOfRequiredOutputs(1);
    this->SetNthOutput(0, this->MakeOutput(0));
}

template <typename TFixedImage, typename TMovingImage>
typename itk::ProcessObject::DataObjectPointer
DemonsRegistration<TFixedImage, TMovingImage>::MakeOutput(DataObjectPointerArraySizeType)
{
    return OutputFieldType::New().GetPointer();
}

template <typename TFixedImage, typename TMovingImage>
void
DemonsRegistration<TFixedImage, TMovingImage>::SetFixedImage(const TFixedImage* image)
{
    this->ProcessObject::SetInput(0, const_cast<TFixedImage*>(image));
}

template <typename TFixedImage, typename TMovingImage>
const TFixedImage*
DemonsRegistration<TFixedImage, TMovingImage>::GetFixedImage() const
{
    return static_cast<const TFixedImage*>(this->ProcessObject::GetInput(0));
}

template <typename TFixedImage, typename TMovingImage>
void
DemonsRegistration<TFixedImage, TMovingImage>::SetMovingImage(const TMovingImage* image)
{
    this->ProcessObject::SetInput(1, const_cast<TMovingImage*>(image));
}

template <typename TFixedImage, typename TMovingImage>
const TMovingImage*
DemonsRegistration<TFixedImage, TMovingImage>::GetMovingImage() const
{
    return static_cast<const TMovingImage*>(this->ProcessObject::GetInput(1));
}

template <typename TFixedImage, typename TMovingImage>
typename DemonsRegistration<TFixedImage, TMovingImage>::OutputFieldType*
DemonsRegistration<TFixedImage, TMovingImage>::GetOutput()
{
    return static_cast<OutputFieldType*>(this->ProcessObject::GetOutput(0));
}

template <typename TFixedImage, typename TMovingImage>
void
DemonsRegistration<TFixedImage, TMovingImage>::GenerateData()
{
    const auto* fixed = this->GetFixedImage();
    const auto* moving = this->GetMovingImage();

    // Demons filters require identical fixed/moving pixel types: cast the
    // moving image to the fixed pixel type.
    typename itk::CastImageFilter<TMovingImage, TFixedImage>::Pointer caster =
        itk::CastImageFilter<TMovingImage, TFixedImage>::New();
    caster->SetInput(const_cast<TMovingImage*>(moving));
    caster->Update();
    typename TFixedImage::ConstPointer movingCast = caster->GetOutput();

    itk::ProcessObject* demons = nullptr;

    switch (m_Variant)
    {
        case DemonsVariant::Classic:
        {
            auto f = itk::DemonsRegistrationFilter<TFixedImage, TFixedImage,
                                                   OutputFieldType>::New();
            f->SetFixedImage(const_cast<TFixedImage*>(fixed));
            f->SetMovingImage(const_cast<TFixedImage*>(movingCast.GetPointer()));
            f->SetNumberOfIterations(m_NumberOfIterations);
            f->SetStandardDeviations(static_cast<float>(m_StandardDeviation));
            f->SetSmoothDisplacementField(m_SmoothDisplacementField);
            demons = f.GetPointer();

            this->AllocateOutputs();
            f->GraftOutput(this->GetOutput());
            f->Update();
            this->GraftOutput(f->GetOutput());
            break;
        }

        case DemonsVariant::Diffeomorphic:
        {
            auto f = itk::DiffeomorphicDemonsRegistrationFilter<TFixedImage, TFixedImage,
                                                                OutputFieldType>::New();
            f->SetFixedImage(const_cast<TFixedImage*>(fixed));
            f->SetMovingImage(const_cast<TFixedImage*>(movingCast.GetPointer()));
            f->SetNumberOfIterations(m_NumberOfIterations);
            f->SetStandardDeviations(static_cast<float>(m_StandardDeviation));
            f->SetMaximumUpdateStepLength(static_cast<float>(m_MaximumUpdateStepLength));
            demons = f.GetPointer();

            this->AllocateOutputs();
            f->GraftOutput(this->GetOutput());
            f->Update();
            this->GraftOutput(f->GetOutput());
            break;
        }

        case DemonsVariant::FastSymmetricForces:
        {
            auto f = itk::FastSymmetricForcesDemonsRegistrationFilter<TFixedImage, TFixedImage,
                                                                      OutputFieldType>::New();
            f->SetFixedImage(const_cast<TFixedImage*>(fixed));
            f->SetMovingImage(const_cast<TFixedImage*>(movingCast.GetPointer()));
            f->SetNumberOfIterations(m_NumberOfIterations);
            f->SetStandardDeviations(static_cast<float>(m_StandardDeviation));
            f->SetSmoothDisplacementField(m_SmoothDisplacementField);
            demons = f.GetPointer();

            this->AllocateOutputs();
            f->GraftOutput(this->GetOutput());
            f->Update();
            this->GraftOutput(f->GetOutput());
            break;
        }
    }

    (void)demons; // kept for potential observer wiring in Phase 1
}

} // namespace medaxis

#endif // medAxisDemonsRegistration_hxx
