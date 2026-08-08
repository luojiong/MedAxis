/*=========================================================================

  medAxisWatershedFilter.hxx

=========================================================================*/
#ifndef medAxisWatershedFilter_hxx
#define medAxisWatershedFilter_hxx

#include "medAxisWatershedFilter.h"

#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkWatershedImageFilter.h"
#include "itkMorphologicalWatershedImageFilter.h"
#include "itkCastImageFilter.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
WatershedFilter<TInputImage, TOutputImage>::WatershedFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
WatershedFilter<TInputImage, TOutputImage>::GenerateData()
{
    using In = TInputImage;
    using Out = TOutputImage;
    using ScalarImageType = itk::Image<float, In::ImageDimension>;

    // Gradient magnitude of the input (used by both modes).
    typename itk::GradientMagnitudeRecursiveGaussianImageFilter<In, ScalarImageType>::Pointer
        gradient = itk::GradientMagnitudeRecursiveGaussianImageFilter<In, ScalarImageType>::New();
    gradient->SetInput(this->GetInput());
    gradient->SetSigma(this->m_GradientSigma);
    gradient->Update();

    if (this->m_Mode == WatershedMode::Gradient)
    {
        // Classic watershed; label type is IdentifierType (unsigned long).
        // We cast labels to the requested output pixel type afterwards.
        auto ws = itk::WatershedImageFilter<ScalarImageType>::New();
        ws->SetInput(gradient->GetOutput());
        ws->SetLevel(this->m_Level);
        ws->SetThreshold(this->m_Threshold);
        ws->Update();

        using LabelImageType = typename itk::WatershedImageFilter<ScalarImageType>::OutputImageType;
        typename itk::CastImageFilter<LabelImageType, Out>::Pointer caster =
            itk::CastImageFilter<LabelImageType, Out>::New();
        caster->SetInput(ws->GetOutput());

        this->AllocateOutputs();
        caster->GraftOutput(this->GetOutput());
        caster->Update();
        this->GraftOutput(caster->GetOutput());
    }
    else // Morphological
    {
        auto ws = itk::MorphologicalWatershedImageFilter<ScalarImageType, Out>::New();
        ws->SetInput(gradient->GetOutput());
        ws->SetLevel(static_cast<float>(this->m_Level));
        ws->SetMarkWatershedLine(this->m_MarkWatershedLine);
        ws->SetFullyConnected(false);

        this->AllocateOutputs();
        ws->GraftOutput(this->GetOutput());
        ws->Update();
        this->GraftOutput(ws->GetOutput());
    }
}

} // namespace medaxis

#endif // medAxisWatershedFilter_hxx
