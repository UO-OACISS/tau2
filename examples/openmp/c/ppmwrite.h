enum base { red, green, blue };

enum direction { back=-1, fwd=1 };

class ColorTable {
public:
  char operator()(int i, base b) const;
  ~ColorTable();
  int numColors() const;
  int shades() const;

protected:
  ColorTable(int shades, int numColors);
  struct Color {
    char c[3];
  };
  Color* tab;

private:
  int    sds;
  int    num;
};

class PermutationColorTable : public ColorTable {
public:
  PermutationColorTable(int shades,
                        direction d=fwd,
                        base b1=red, base b2=green, base b3=blue);
};

class SmoothColorTable : public ColorTable {
public:
  SmoothColorTable(int shades,
                   direction d=fwd,
                   base b1=red, base b2=green, base b3=blue);
};

const int width=800, height=800;     // Resolution
typedef int field[width][height];    // Type to store iteration counts

void ppmwrite(char *fname,           // Name of PPM file
              field iterations,      // Calculated iteration counts
              int maxiter,           // Iteration count limit
              const ColorTable& table=SmoothColorTable(10, fwd, blue, green, red));
                                     // Default color table
