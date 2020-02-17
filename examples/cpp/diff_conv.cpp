#include <taichi/util.h>

TI_NAMESPACE_BEGIN

auto diff_conv = [](const std::vector<std::string> args) {
  int grid_resolution = 254;
  TI_ASSERT(args.size() == 3);
  float th = std::stof(args[2]);
  TI_P(th);
  auto f = fopen(args[0].c_str(), "rb");

  int n = pow<3>(grid_resolution);
  TI_ASSERT(f);

  std::vector<float32> ret1(n);
  trash(std::fread(ret1.data(), sizeof(float32), ret1.size(), f));
  std::fclose(f);

  f = fopen(args[1].c_str(), "rb");
  TI_ASSERT(f);
  std::vector<float32> ret2(n);
  trash(std::fread(ret2.data(), sizeof(float32), ret2.size(), f));
  std::fclose(f);

  int counter[2] = {0, 0};
  double sum1 = 0, sum2 = 0;
  float max1 = 0, max2 = 0;
  int non_zero1 = 0, non_zero2 = 0;
  int total_non_zero = 0;
  for (int i = 0; i < n; i++) {
    sum1 += std::abs(ret1[i]);
    sum2 += std::abs(ret2[i]);
    max1 = std::max(max1, ret1[i]);
    max2 = std::max(max2, ret1[i]);
    bool same = std::abs(ret1[i] - ret2[i]) < th;
    bool non_zero = (ret1[i] != 0) || (ret2[i] != 0);
    total_non_zero += non_zero;
    if (same)
      counter[0]++;

    if (same && total_non_zero)
      counter[1]++;

    if (ret1[i] == 0) {
      non_zero1++;
    }

    if (ret2[i] == 0) {
      non_zero2++;
    }

    // if (!same) {
    // fprintf(stderr, "ret1:%f, ret2:%f\n", ret1[i], ret2[i]);
    //}
  }
  TI_INFO("same {} {}%", counter[0], 100.0f * counter[0] / n);
  TI_INFO("non zero same {} {}%", counter[0],
          100.0f * counter[1] / total_non_zero);
  TI_P(sum1 / n);
  TI_P(sum2 / n);
  TI_P(sum1 / total_non_zero);
  TI_P(sum2 / total_non_zero);
  TI_P(max1);
  TI_P(max2);
  TI_P(non_zero1);
  TI_P(non_zero2);
};

TI_REGISTER_TASK(diff_conv);

TI_NAMESPACE_END
