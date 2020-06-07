#define MAX_MESSAGES (1024 * 4)  // * 4 * 32 = 512 KB
#define MSG_SIZE 32
// 2 left for the `type` bitmap, 1 left for the contents-count
#define MAX_CONTENTS_PER_MSG (MSG_SIZE - 3)

#ifdef __GLSL__
// clang-format off

#include "taichi/util/macros.h"
STR(
struct _msg_entry_t {
  int contents[29];
  int num_contents;
  int type_bm_lo;
  int type_bm_hi;
};

layout(std430, binding = 6) buffer runtime {
  int _rand_state_;
  int _msg_count_;
  _msg_entry_t _msg_buf_[];
};
)

// clang-format on
#else

struct GLSLMsgEntry {
  union MsgValue {
    int32 val_i32;
    float32 val_f32;
  } contents[MAX_CONTENTS_PER_MSG];

  int num_contents;
  int type_bm_lo;
  int type_bm_hi;

  int get_type_of(int i) const {
    int type = (type_bm_lo >> i) & 1;
    type |= ((type_bm_hi >> i) & 1) << 1;
    return type;
  }
} __attribute__((packed));

struct GLSLRuntime {
  int rand_state;
  int msg_count;
  GLSLMsgEntry msg_buf[MAX_MESSAGES * MSG_SIZE];
} __attribute__((packed));

#endif
