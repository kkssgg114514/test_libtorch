#include "SpeakerDataSet.h"

SpeakerDataSet::SpeakerDataSet(const std::string& feature_dir, int feature_dim, int chunk_size, bool is_train)
{
}

int64_t SpeakerDataSet::num_speakers() const
{
	return 0;
}

torch::optional<size_t> SpeakerDataSet::size() const
{
	return torch::optional<size_t>();
}

SpeakerSample SpeakerDataSet::get(size_t index)
{
	return SpeakerSample();
}
