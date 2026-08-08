/*=========================================================================

  medAxisRegionGrowFilter.hxx

=========================================================================*/
#ifndef medAxisRegionGrowFilter_hxx
#define medAxisRegionGrowFilter_hxx

#include "medAxisRegionGrowFilter.h"

#include "itkConnectedThresholdImageFilter.h"
#include "itkConfidenceConnectedImageFilter.h"
#include "itkNeighborhoodConnectedImageFilter.h"
#include "itkCastImageFilter.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
RegionGrowFilter<TInputImage, TOutputImage>::RegionGrowFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
RegionGrowFilter<TInputImage, TOutputImage>::GenerateData()
{
    if (this->m_Seeds.empty())
    {
        itkExceptionMacro("RegionGrowFilter: no seed points provided.");
    }

    using In = TInputImage;
    using Out = TOutputImage;
    using InternalPixelType = double;
    using InternalImageType = itk::Image<InternalPixelType, In::ImageDimension>;

    // Run internally on double to keep thresholds independent of pixel type.
    typename itk::CastImageFilter<In, InternalImageType>::Pointer caster =
        itk::CastImageFilter<In, InternalImageType>::New();
    caster->SetInput(this->GetInput());
    caster->Update();

    typename itk::ImageToImageFilter<InternalImageType, Out>::Pointer grower;

    switch (this->m_Mode)
    {
        case RegionGrowMode::Connected:
        {
            auto f = itk::ConnectedThresholdImageFilter<InternalImageType, Out>::New();
            f->SetLower(this->m_LowerThreshold);
            f->SetUpper(this->m_UpperThreshold);
            f->SetReplaceValue(this->m_ReplaceValue);
            for (const auto& seed : this->m_Seeds)
            {
                f->AddSeed(seed);
            }
            grower = f;
            break;
        }

        case RegionGrowMode::Confidence:
        {
            auto f = itk::ConfidenceConnectedImageFilter<InternalImageType, Out>::New();
            f->SetMultiplier(this->m_Multiplier);
            f->SetNumberOfIterations(this->m_Iterations);
            f->SetReplaceValue(this->m_ReplaceValue);
            f->SetInitialNeighborhoodRadius(this->m_NeighborhoodRadius);
            for (const auto& seed : this->m_Seeds)
            {
                f->AddSeed(seed);
            }
            grower = f;
            break;
        }

        case RegionGrowMode::Neighborhood:
        {
            auto f = itk::NeighborhoodConnectedImageFilter<InternalImageType, Out>::New();
            f->SetLower(this->m_LowerThreshold);
            f->SetUpper(this->m_UpperThreshold);
            f->SetReplaceValue(this->m_ReplaceValue);
            typename itk::NeighborhoodConnectedImageFilter<InternalImageType, Out>::RadiusType r;
            r.Fill(this->m_NeighborhoodRadius);
            f->SetRadius(r);
            for (const auto& seed : this->m_Seeds)
            {
                f->AddSeed(seed);
            }
            grower = f;
            break;
        }
    }

    grower->SetInput(caster->GetOutput());

    this->AllocateOutputs();
    grower->GraftOutput(this->GetOutput());
    grower->Update();
    this->GraftOutput(grower->GetOutput());
}

} // namespace medaxis

#endif // medAxisRegionGrowFilter_hxx
