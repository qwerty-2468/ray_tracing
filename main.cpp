//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Vizhanyo Miklos Ferenc
// Neptun : NVY1AG
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const float epsilon = 0.0001f;

enum MaterialType { ROUGH, REFLECTIVE, REFRACTIVE };

struct Material {
    vec3 ka, kd, ks;
    float shininess;
    vec3 F0;
    float ior;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * 3;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
    }
};

struct RefractiveMaterial : Material {
    RefractiveMaterial(vec3 n) : Material(REFRACTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one) * (n - one)) / ((n + one) * (n + one));
        ior = n.x;
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    bool out;
    Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out=_out;}
};

class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
    vec3 center;
    float radius;
public:
    Sphere(const vec3& _center, float _radius, Material* _material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }
};

struct Plane : public Intersectable {
    vec3 p;
    vec3 n;
    float width;
    Material* material1;
    Material* material2;

public:
    Plane(const vec3& _p, const vec3& _n, float _width, Material* _material1, Material* _material2) {
        p = _p; n = normalize(_n); width = _width; material1 = _material1; material2 = _material2;
    }
    Hit intersect(const Ray& ray) {
        Hit hit;

        float d = dot(ray.dir,n);
        if (fabs(d) < epsilon) return hit;
        float t = -dot(ray.start-p, n) / d;

        if (t <= 0) return hit;
        vec3 position = ray.start + ray.dir * t;
        vec3 toCenter = position - p;
        float halfWidth = width / 2.0f;
        if (fabs(dot(toCenter, cross(n, vec3(0, 0, 1)))) <= halfWidth && fabs(dot(toCenter, cross(n, vec3(1, 0, 0)))) <= halfWidth) {
            int u = static_cast<int>((toCenter.x + halfWidth));
            int v = static_cast<int>((toCenter.z + halfWidth));
            Material* material = ((u + v) % 2 == 0) ? material2 : material1;
            hit.t = t;
            hit.position = position;
            hit.normal = n;
            hit.material = material;
        }
        return hit;
    }
};

struct Cone : public Intersectable {
    vec3 p, n;
    float h, alpha;
public:
    Cone(const vec3& _p, vec3 _n, float _alpha, float _h, Material* _material) {
        p = _p;
        n = normalize(_n);
        alpha = _alpha;
        h = _h;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        float a = pow(dot(ray.dir,n),2) - dot(ray.dir, ray.dir)*pow(cos(alpha), 2);
        float b = 2*dot(ray.dir,n)*dot(ray.start-p,n) - 2*pow(cos(alpha),2)*dot(ray.start-p, ray.dir);
        float c = pow(dot(ray.start-p, n), 2) - dot(ray.start-p,ray.start-p)*pow(cos(alpha),2);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        float t;
        if(t1<=0){
            if(t2<=0) return hit;
            else t = t2;
        } else {
            if(t2<=0) t = t1;
            else t = t1<t2 ? t1 : t2;
        }
        vec3 position = ray.start + ray.dir * t;
        vec3 r = position;
        if(dot(position-p,n)>h || dot(position-p,n)<0) return hit;
        hit.t=t;
        hit.position=position;
        hit.normal = normalize(2*(dot(r-p, n)*n) - 2*(r-p)*pow(cos(alpha), 2));
        hit.material = material;
        return hit;
    }
};

struct Cylinder : public Intersectable {
    vec3 p, v0;
    float R, h;
    public:
        Cylinder(const vec3& _p, const vec3& _v0, float _R, float _h, Material* _material) {
        p = _p;
        v0 = normalize(_v0);
        R = _R;
        h = _h;
        material = _material;
    }

        Hit intersect(const Ray& ray) {
        Hit hit;
        float a = dot(ray.dir,ray.dir) - 2*dot(ray.dir,v0*dot(ray.dir,v0)) + dot(dot(ray.dir,v0)*v0,dot(ray.dir,v0)*v0);
        float b = 2*dot(ray.start-p, ray.dir) - 2*dot(ray.dir,dot(ray.start-p,v0)*v0) - 2*dot(ray.start-p,dot(ray.dir,v0)*v0) + 2*dot(ray.dir,dot(ray.start-p,v0)*v0);
        float c = dot(ray.start-p,ray.start-p) - 2*(dot(ray.start-p,dot(ray.start-p,v0)*v0)) + dot(dot(ray.start-p,v0)*v0,dot(ray.start-p,v0)*v0) - pow(R,2);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        float t;
        if(t1<=0){
            if(t2<=0) return hit;
            else t = t2;
        } else {
            if(t2<=0) t = t1;
            else t = t1<t2 ? t1 : t2;
        }
        vec3 position = ray.start + ray.dir * t;
        vec3 r = position;
        if(dot(r-p,v0)<0 || dot(r-p,v0)>h) return hit;
        hit.t=t;
        hit.position=position;
        hit.normal = normalize(r-p-v0*dot(r-p,v0));
        hit.material = material;
        return hit;
    }
};



class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float windowSize = length(w) * tanf(fov / 2);
        right = normalize(cross(vup, w)) * windowSize;
        up = normalize(cross(w, right)) * windowSize;
    }

    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir, true);
    }

    void Animate(float dt) {
        vec3 d = eye - lookat;
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
        set(eye, lookat, vec3(0,1,0), fov);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(0, 1, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.4f, 0.4f, 0.4f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        objects.push_back(new Plane(vec3(0, -1, 0), vec3(0, -1, 0), 20,
            new RoughMaterial(vec3(0, 0.1, 0.3), vec3(0,0,0), 0),
            new RoughMaterial(vec3(0.3, 0.3, 0.3), vec3(0,0,0), 0)));
        objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0),0.3, 2,
            new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9))));
        objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 0.3, 2,
            new RefractiveMaterial(vec3(1.3,1.3,1.3))));
        objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 0.3, 2,
            new RoughMaterial(vec3(0.3, 0.2, 0.1), vec3(2, 2, 2), 50)));
        objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 0.2, 2,
            new RoughMaterial(vec3(0.1, 0.2, 0.3), vec3(2, 2, 2), 100)));
        objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 0.2, 2,
            new RoughMaterial(vec3(0.3, 0, 0.2), vec3(2, 2, 2), 20)));
    }

    void render(std::vector<vec4>& image) {
        long timeStart = glutGet(GLUT_ELAPSED_TIME);

        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
        printf("Rendering time: %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable* object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {
        for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }
    
    vec3 refract(vec3 V, vec3 N, float ns) {
        float cosa = -dot(V, N);
        float disc = 1 - (1 - cosa*cosa)/ns/ns;
        if (disc < 0) return vec3(0, 0, 0);
        return V/ns + N * (cosa/ns - sqrt(disc));
    }
    
    vec3 Fresnel(vec3 V, vec3 N, vec3 n, vec3 kappa) {
        float cosa = -dot(V, N);
        vec3 one(1, 1, 1);
        vec3 F0 = ((n-one)*(n-one) + kappa*kappa) /
        ((n+one)*(n+one) + kappa*kappa);
        return F0 + (one-F0) * pow(1-cosa, 5);
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        
        if (hit.material->type == ROUGH) {
            vec3 outRadiance = hit.material->ka * La;
            for (Light* light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->direction, true);
                float cosTheta = dot(hit.normal, light->direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
                    outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light->direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                }
            }
            return outRadiance;
        }

        float cosa = -dot(ray.dir, hit.normal);
        vec3 one(1, 1, 1);
        vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
        vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
        vec3 outRadiance = trace(Ray(hit.position + hit.normal * epsilon, reflectedDir, true), depth + 1) * F;
        
        if (hit.material->type == REFRACTIVE) {
            float ior = (ray.out) ? hit.material->ior : 1/hit.material->ior;
            vec3 refractionDir = refract(ray.dir,hit.normal,ior);
            if (length(refractionDir) > 0) {
                Ray refractRay(hit.position-hit.normal*epsilon, refractionDir, !ray.out);
                outRadiance = outRadiance + trace(refractRay, depth+1) * (vec3(1,1,1)-Fresnel(ray.dir, hit.normal, vec3(hit.material->ior,hit.material->ior,hit.material->ior), vec3(0,0,0)));
            }
        }
        return outRadiance;
    }

    void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram;
Scene scene;

const char* vertexSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec2 cVertexPosition;
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
    }
)";

const char* fragmentSource = R"(
    #version 330
    precision highp float;

    uniform sampler2D textureUnit;
    in  vec2 texcoord;
    out vec4 fragmentColor;

    void main() {
        fragmentColor = texture(textureUnit, texcoord);
    }
)";

class FullScreenTexturedQuad {
    unsigned int vao = 0, textureId = 0;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void LoadTexture(std::vector<vec4>& image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() {
        glBindVertexArray(vao);
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad->LoadTexture(image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    if(key == 'a'){
        scene.Animate(2*M_PI/8);
        glutPostRedisplay();
    }
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}
