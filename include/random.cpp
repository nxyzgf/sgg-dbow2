#include "Random.h"
#include "Timestamp.h"
#include <cstdlib>

using namespace std;

namespace DBoW2 {
	bool Random::m_already_seeded = false;

	void Random::SeedRand() {
		Timestamp time;
		time.setToCurrentTime();
		srand((unsigned)time.getFloatTime());
	}

	void Random::SeedRandOnce()
	{
		if (!m_already_seeded)
		{
			Random::SeedRand();
			m_already_seeded = true;
		}
	}

	void Random::SeedRand(int seed)
	{
		srand(seed);
	}

	void Random::SeedRandOnce(int seed)
	{
		if (!m_already_seeded)
		{
			Random::SeedRand(seed);
			m_already_seeded = true;
		}
	}

	int Random::RandomInt(int min, int max) {
		int d = max - min + 1;
		return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
	}

	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------

	Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
	{
		if (min <= max)
		{
			m_min = min;
			m_max = max;
		}
		else
		{
			m_min = max;
			m_max = min;
		}

		createValues();
	}

	// ---------------------------------------------------------------------------

	Random::UnrepeatedRandomizer::UnrepeatedRandomizer
	(const Random::UnrepeatedRandomizer& rnd)
	{
		*this = rnd;
	}

	// ---------------------------------------------------------------------------

	int Random::UnrepeatedRandomizer::get()
	{
		if (empty()) createValues();

		Random::SeedRandOnce();

		int k = Random::RandomInt(0, m_values.size() - 1);
		int ret = m_values[k];
		m_values[k] = m_values.back();
		m_values.pop_back();

		return ret;
	}

	// ---------------------------------------------------------------------------

	void Random::UnrepeatedRandomizer::createValues()
	{
		int n = m_max - m_min + 1;

		m_values.resize(n);
		for (int i = 0; i < n; ++i) m_values[i] = m_min + i;
	}

	// ---------------------------------------------------------------------------

	void Random::UnrepeatedRandomizer::reset()
	{
		if ((int)m_values.size() != m_max - m_min + 1) createValues();
	}

	// ---------------------------------------------------------------------------

	Random::UnrepeatedRandomizer&
		Random::UnrepeatedRandomizer::operator=
		(const Random::UnrepeatedRandomizer& rnd)
	{
		if (this != &rnd)
		{
			this->m_min = rnd.m_min;
			this->m_max = rnd.m_max;
			this->m_values = rnd.m_values;
		}
		return *this;
	}

}