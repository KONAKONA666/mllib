#pragma once

namespace constants {
	constexpr int BATCH_SIZE = 1000;
	constexpr int BLOCK_WIDTH = 3000;
	constexpr int THREADS = 256;
	constexpr int LABELS = 10;
	constexpr int SEED = 42;
	constexpr int TRAIN_SIZE = 50000;
	constexpr int TEST_SIZE = 10000;
	constexpr float learningRate = 0.001f;
}