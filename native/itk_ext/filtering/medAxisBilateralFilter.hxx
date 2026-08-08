/*=========================================================================

  medAxisBilateralFilter.hxx

=========================================================================*/
#ifndef medAxisBilateralFilter_hxx
#define medAxisBilateralFilter_hxx

#include "medAxisBilateralFilter.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
BilateralFilter<TInputImage, TOutputImage>::BilateralFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
BilateralFilter<TInputImage, TOutputImage>::GenerateData()
{
    using InternalFilterType = itk::BilateralImageFilter<TInputImage, TOutputImage>;
    typename InternalFilterType::Pointer internal = InternalFilterType::New();

    internal->SetInput(this->GetInput());
    internal->SetDomainSigma(this->m_DomainSigma);
    internal->SetRangeSigma(this->m_RangeSigma);
    internal->SetNumberOfRangeGaussianSamples(this->m_NumberOfRangeGaussianSamples);

    // Graft our output into the internal filter so it fills our buffer.
    this->AllocateOutputs();
    internal->GraftOutput(this->GetOutput());
    internal->Update();
    this->GraftOutput(internal->GetOutput());
}

} // namespace medaxis

#endif // medAxisBilateralFilter_hxx
