#include <math.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/glut.h>

#define WINDOW_WIDTH 364
#define WINDOW_HEIGHT 364
#define FBO_WIDTH 1024
#define FBO_HEIGHT 1024
#define NUM_LIGHT 1
#define WINDOW_ASPECT ((float)WINDOW_WIDTH/(float)WINDOW_HEIGHT)

const GLchar* PASS1_VERT_SHADER =
    "#version 130\n"
    "uniform mat4 in_MVP;\n"
    "uniform mat4 in_MV;\n"
    "uniform mat4 in_Normal_MV;\n" // ModelView行列から平行移動を除いた回転行列のみの行列
    "attribute vec4 in_Position;\n"
    "attribute vec4 in_Normal;\n"
    "attribute vec2 in_Texture_coord;\n"
    "varying vec4 v_Position;\n"
    "varying vec4 v_Normal;\n"
    "varying vec2 v_texture_coord;\n"
    "void main(void)\n"
    "{\n"
    "    gl_Position = in_MVP * in_Position;\n"
    "    v_Position = in_MV * in_Position;\n"
    "    v_Normal = in_Normal_MV * in_Normal;\n"
    "    v_texture_coord = in_Texture_coord;\n"
    "}\n";

const GLchar* PASS1_FRAG_SHADER =
    "#version 130\n"
    "precision highp float;\n"
    "uniform sampler2D in_Img;\n"
    "varying vec4 v_Position;\n"
    "varying vec4 v_Normal;\n"
    "varying vec2 v_texture_coord;\n"
    "void main(void)\n"
    "{\n"
    "    gl_FragData[0] = v_Position;\n"
    "    gl_FragData[1] = v_Normal;\n"
    "    gl_FragData[2] = texture2D(in_Img, v_texture_coord);\n"
    "}\n";

const GLchar* PASS2_VERT_SHADER =
    "#version 130\n"
    "attribute vec2 in_Position;\n"
    "attribute vec2 in_Texture_coord;\n"
    "varying vec2 v_texture_coord;\n"
    "void main(void)\n"
    "{\n"
    "    gl_Position = vec4(in_Position, 0.0, 1.0);\n"
    "    v_texture_coord = in_Texture_coord;\n"
    "}\n";

#define STR(str) DOSTR(str)
#define DOSTR(str) # str
const GLchar* PASS2_FRAG_SHADER =
    "#version 130\n"
    "precision highp float;\n"
    "uniform sampler2D in_Position_Img;\n"
    "uniform sampler2D in_Normal_Img;\n"
    "uniform sampler2D in_Albedo_Img;\n"
    "uniform vec3 in_light_pos[" STR(NUM_LIGHT) "];\n" // ライトの座標
    "uniform vec3 in_light_power[" STR(NUM_LIGHT) "];\n" // ライトの出力
    "uniform float in_light_dist[" STR(NUM_LIGHT) "];\n" // スポットライトの減衰開始距離
    "varying vec2 v_texture_coord;\n"
    "void main(void)\n"
    "{\n"
    "    vec4 pos4 = texture2D(in_Position_Img, v_texture_coord);\n"
    "    if (pos4.w <= 0.0) discard;\n" // glClearで初期化されただけの場所は描画しない
    "    vec3 pos = pos4.xyz;\n"
    "    vec3 normal = normalize(texture2D(in_Normal_Img, v_texture_coord).xyz);\n"
    "    vec3 albedo = texture2D(in_Albedo_Img, v_texture_coord).xyz;\n"
    "    vec3 frag_color = vec3(0.0, 0.0, 0.0);\n"
    "    for (int i = 0; i < " STR(NUM_LIGHT) "; i++) {\n"
    "        vec3 dist = (in_light_pos[i] - pos);\n"
    "        vec3 dir = normalize(dist);\n"
    "        float dir_power = min(1.0, max(0.0, dot(dir, normal)));\n"
    "        float len_power = 1.0 / pow(max(1.0, length(dist) / in_light_dist[i]),2.0);\n" // ポイントライトの減衰率(逆2乗)
    "        frag_color += (albedo*in_light_power[i])*(dir_power*len_power);\n" // 拡散反射のみ計算
    "    }\n"
    "    gl_FragColor = vec4(frag_color, 1.0);\n"
    "}\n";

static GLuint gPass1Program;
static GLuint gPass2Program;
static GLuint gPositionTexture;
static GLuint gNormalTexture;
static GLuint gAlbedoTexture; // "albedo" means texture color
static GLuint gImg;
static GLuint gFrameBufferObject;

struct Lights {
    float pos[3*NUM_LIGHT];
    float power[3*NUM_LIGHT];
    float dist[NUM_LIGHT];
};

static Lights gLights;

/**
 * geometory to texture
 */
static void draw_pass1() {
    glUseProgram(gPass1Program);

    glViewport(0,0,FBO_WIDTH,FBO_HEIGHT);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, gFrameBufferObject);

    static const GLenum bufs[] = {
      GL_COLOR_ATTACHMENT0_EXT,
      GL_COLOR_ATTACHMENT1_EXT,
      GL_COLOR_ATTACHMENT2_EXT,
    };
    glDrawBuffers(3, bufs);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0,0,0,0); // w値を0にしておくことで、ポリゴンのフラグメントのみをシャーダで処理できる
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(gPass1Program, "in_Img"), 0);
    glBindTexture(GL_TEXTURE_2D, gImg);

    const float vertexPointer[] = {
            0.f,   2.0f, 0.f,
           -2.0f, -2.0f, 0.f,
            2.0f, -2.0f, 0.f,
        };

    const float normalPointer[] = {
            0.f, 0.f, 1.f,
            0.f, 0.f, 1.f,
            0.f, 0.f, 1.f,
        };

    const float texturePointer[] = {
            0.5f, 1.f,
            0.f, 0.f,
            1.f, 0.f,
        };

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glVertexAttribPointer(0, 3, GL_FLOAT, 0, sizeof (GLfloat) * 3, vertexPointer);
    glVertexAttribPointer(1, 3, GL_FLOAT, 0, sizeof (GLfloat) * 3, normalPointer);
    glVertexAttribPointer(2, 2, GL_FLOAT, 0, sizeof (GLfloat) * 2, texturePointer);

    glDrawArrays(GL_TRIANGLES,0,3);
    glFlush();

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    glUseProgram(0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glDrawBuffer(GL_FRONT);

    int err = glGetError();
    if (GL_NO_ERROR != err) {
        printf("Check GL Error on display() pass1: %d\n", err);
    }
}

/**
 * extract geometory from texture. and render using it.
 */
static void draw_pass2() {
    glUseProgram(gPass2Program);
    glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glUniform1i(glGetUniformLocation(gPass2Program, "in_Position_Img"), 0);
    glUniform1i(glGetUniformLocation(gPass2Program, "in_Normal_Img"), 1);
    glUniform1i(glGetUniformLocation(gPass2Program, "in_Albedo_Img"), 2);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPositionTexture);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormalTexture);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gAlbedoTexture);

    glUniform3fv(glGetUniformLocation(gPass2Program, "in_light_pos"),
        NUM_LIGHT, gLights.pos);
    glUniform3fv(glGetUniformLocation(gPass2Program, "in_light_power"),
        NUM_LIGHT, gLights.power);
    glUniform1fv(glGetUniformLocation(gPass2Program, "in_light_dist"),
        NUM_LIGHT, gLights.dist);

    const float vertexPointer[] = {
            -1.0,  1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0, -1.0
        };

    const float texturePointer[] = {
            0.0,  1.0,
            1.0,  1.0,
            0.0,  0.0,
            1.0,  0.0
        };

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(2);

    glVertexAttribPointer(0, 2, GL_FLOAT, 0, sizeof (GLfloat) * 2, vertexPointer);
    glVertexAttribPointer(2, 2, GL_FLOAT, 0, sizeof (GLfloat) * 2, texturePointer);

    glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    glFlush();

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(2);
}

static void draw_texture(GLuint textureId) {
    glUseProgram(0);

    glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);

    glEnable(GL_TEXTURE_2D);

    glColor3d(1.0, 1.0, 1.0);
    glBegin(GL_TRIANGLE_STRIP);
      glTexCoord2d(0.0, 1.0);
      glVertex2d(-1.0,  1.0);
      
      glTexCoord2d(1.0, 1.0);
      glVertex2d( 1.0,  1.0);

      glTexCoord2d(0.0, 0.0);
      glVertex2d(-1.0, -1.0);

      glTexCoord2d(1.0, 0.0);
      glVertex2d( 1.0, -1.0);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    int err = glGetError();
    if (GL_NO_ERROR != err) {
        printf("Check GL Error on display() pass2: %d\n", err);
    }
}

static void display(void) {
    draw_pass1();
    draw_pass2();
    
    //draw_texture(gAlbedoTexture);
    //draw_texture(gNormalTexture);
    
    glFlush();

    int err = glGetError();
    if (GL_NO_ERROR != err) {
        printf("Check GL Error on display(): %d\n", err);
    }
}

static void idle(void) {
    glutPostRedisplay();
}

static void printShaderInfoLog(GLuint shader) {
    GLsizei bufSize;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH , &bufSize);
    if (bufSize > 1) {
        GLchar *infoLog;
        infoLog = (GLchar *)malloc(bufSize);
        if (infoLog != NULL) {
            GLsizei length;
            glGetShaderInfoLog(shader, bufSize, &length, infoLog);
            fprintf(stderr, "InfoLog:\n%s\n\n", infoLog);
            free(infoLog);
        } else {
            fprintf(stderr, "Could not allocate InfoLog buffer.\n");
        }
    }
}

static void printProgramInfoLog(GLuint program) {
    GLsizei bufSize;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH , &bufSize);
    if (bufSize > 1) {
        GLchar *infoLog = (GLchar *)malloc(bufSize);
        if (infoLog != NULL) {
            GLsizei length;
            glGetProgramInfoLog(program, bufSize, &length, infoLog);
            fprintf(stderr, "InfoLog:\n%s\n\n", infoLog);
            free(infoLog);
        } else {
            fprintf(stderr, "Could not allocate InfoLog buffer.\n");
        }
    }
}

static GLuint loadShader(const GLchar* vertSource, const GLchar* fragSource, int pass) {
    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);

    int length = strlen(vertSource)+1;
    glShaderSource(vert, 1, &vertSource, &length);

    length = strlen(fragSource)+1;
    glShaderSource(frag, 1, &fragSource, &length);

    GLint compiled, linked;

    glCompileShader(vert);
    glGetShaderiv(vert, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        printShaderInfoLog(vert);
        exit(1);
    }

    glCompileShader(frag);
    glGetShaderiv(frag, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        printShaderInfoLog(frag);
        exit(1);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);

    glDeleteShader(vert);
    glDeleteShader(frag);

    if (pass == 1) {
        glBindAttribLocation(program, 0, "in_Position");
        glBindAttribLocation(program, 1, "in_Normal");
        glBindAttribLocation(program, 2, "in_Texture_coord");
    } else if (pass == 2) {
        glBindAttribLocation(program, 0, "in_Position");
        glBindAttribLocation(program, 2, "in_Texture_coord");
    }

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        printProgramInfoLog(program);
        exit(1);
    }

    return program;
}

static void multiplyMatrix(float* out, const float* src1, const float* src2) {
    for (int i=0; i<16; i++) {
        out[i] = 0;
        const int src1_base = (i/4)*4; // 0->4->8->12
        const int src2_base = i%4;
        for (int j=0; j<4; j++) {
            const int src1_index = src1_base + j;
            const int src2_index = src2_base + j*4;
            out[i] += (src1[src1_index]*src2[src2_index]);
        }
    }
}

static void getPerspectiveMatrix(float* proj, float aspect,
        int fovy, float near, float far) {
    const float f = 1.0f / (float) tan(fovy * (M_PI / 360.0f));
    const float rangeReciprocal = 1.0f / (near - far);

    proj[0] = f / aspect;
    proj[1] = 0.0f;
    proj[2] = 0.0f;
    proj[3] = 0.0f;

    proj[4] = 0.0f;
    proj[5] = f;
    proj[6] = 0.0f;
    proj[7] = 0.0f;

    proj[8] = 0.0f;
    proj[9] = 0.0f;
    proj[10] = (far + near) * rangeReciprocal;
    proj[11] = -1.0f;

    proj[12] = 0.0f;
    proj[13] = 0.0f;
    proj[14] = 2.0f * far * near * rangeReciprocal;
    proj[15] = 0.0f;
}

static void getOrthoMatrix(float* proj, float left, float right,
        float top, float bottom, float near, float far) {
    const float tx = -((right+left)/(right-left));
    const float ty = -((top+bottom)/(top-bottom));
    const float tz = -((far+near)/(far-near));

    proj[0] = 2.f/(right-left);
    proj[1] = 0.f;
    proj[2] = 0.f;
    proj[3] = 0.f;

    proj[4] = 0.f;
    proj[5] = 2.f/(top-bottom);
    proj[6] = 0.f;
    proj[7] = 0.f;

    proj[8] = 0.f;
    proj[9] = 0.f;
    proj[10] = -2.f/(far-near);
    proj[11] = 0.f;

    proj[12] = tx;
    proj[13] = ty;
    proj[14] = tz;
    proj[15] = 1.f;
}

static void getModelviewMatrix(float* modelview, float* normalview,
        float ex, float ey, float ez,
        float ax, float ay, float az,
        float ux, float uy, float uz) {
    float fx = ax - ex;
    float fy = ay - ey;
    float fz = az - ez;

    // Normalize f
    float rlf = 1.0f / sqrtf(fx * fx + fy * fy + fz * fz);
    fx *= rlf;
    fy *= rlf;
    fz *= rlf;

    // compute s = f x_ up (x_ means "cross product")
    float sx = fy * uz - fz * uy;
    float sy = fz * ux - fx * uz;
    float sz = fx * uy - fy * ux;

    // and normalize s
    float rls = 1.0f / sqrtf(sx * sx + sy * sy + sz * sz);
    sx *= rls;
    sy *= rls;
    sz *= rls;

    // compute u = s x_ f
    float upx = sy * fz - sz * fy;
    float upy = sz * fx - sx * fz;
    float upz = sx * fy - sy * fx;

    const float rote[16] = {
            sx, upx, -fx, 0,
            sy, upy, -fy, 0,
            sz, upz, -fz, 0,
             0,   0,   0, 1
        };
    if (normalview) memcpy(normalview, rote, sizeof(rote));

    const float move[16] = {
             1,   0,   0, 0,
             0,   1,   0, 0,
             0,   0,   1, 0,
           -ex, -ey, -ez, 1
        };

    multiplyMatrix(modelview, rote, move);
}

static void initPass1Shader() {
    glUseProgram(gPass1Program);

    float mvp[16];
    float proj[16];
    float modelview[16];
    float normalview[16];

    getPerspectiveMatrix(proj, 1.f, 60, 0.1f, 100.f);
    getModelviewMatrix(modelview, normalview, 0,0,5.5f, 0,0,0, 0,1,0);
    multiplyMatrix(mvp, modelview, proj);

    glUniformMatrix4fv(glGetUniformLocation(gPass1Program, "in_MVP"), 1, GL_FALSE, mvp);
    glUniformMatrix4fv(glGetUniformLocation(gPass1Program, "in_MV"), 1, GL_FALSE, modelview);
    glUniformMatrix4fv(glGetUniformLocation(gPass1Program, "in_Normal_MV"), 1, GL_FALSE, normalview);

    // Positionテクスチャの用意
    glGenTextures(1, &gPositionTexture);
    glBindTexture(GL_TEXTURE_2D, gPositionTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FBO_WIDTH, FBO_HEIGHT, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Normalテクスチャの用意
    glGenTextures(1, &gNormalTexture);
    glBindTexture(GL_TEXTURE_2D, gNormalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FBO_WIDTH, FBO_HEIGHT, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Albedoテクスチャの用意
    glGenTextures(1, &gAlbedoTexture);
    glBindTexture(GL_TEXTURE_2D, gAlbedoTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, FBO_WIDTH, FBO_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // テクスチャマッピングのテクスチャを用意(単純画像のため1画素のみ)
    const uint8_t img[] = {255, 255, 255, 255};
    glGenTextures(1, &gImg);
    glBindTexture(GL_TEXTURE_2D, gImg);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // テクスチャをフレームバッファに関連付ける
    glGenFramebuffersEXT(1, &gFrameBufferObject);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, gFrameBufferObject);

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, gPositionTexture, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, gNormalTexture, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, gAlbedoTexture, 0);

    GLuint depthrenderbuffer;
    glGenRenderbuffersEXT(1, &depthrenderbuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthrenderbuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, FBO_WIDTH, FBO_HEIGHT);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthrenderbuffer);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Failed to initialize FBO\n");
        exit(1);
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    // TODO ライトの初期化
    gLights.pos[0] = 0.9;
    gLights.pos[1] = 0.0;
    gLights.pos[2] = 1.0;

    gLights.power[0] = 1.f;
    gLights.power[1] = 1.f;
    gLights.power[2] = 1.f;

    gLights.dist[0] = 2.3;
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(display);
    glutIdleFunc(idle);

    int err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Failed to initialize glew: %d\n", err);
        exit(1);
    }

    gPass1Program = loadShader(PASS1_VERT_SHADER, PASS1_FRAG_SHADER, 1);
    initPass1Shader();

    gPass2Program = loadShader(PASS2_VERT_SHADER, PASS2_FRAG_SHADER, 2);

    glutMainLoop();
    return 0;
}

