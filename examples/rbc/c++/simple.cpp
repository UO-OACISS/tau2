#include <iostream>

using namespace std;

#define DATA_COUNT 10
#define OVERRUN    5

int main(int argc, char ** argv)
{
  int i;
  int * data;

  double * x = new double;

  data = new int[DATA_COUNT];

  for(i=0; i<DATA_COUNT+OVERRUN; ++i) {
    cout << "Setting data[" << i << "] to " << i << endl;
    data[i] = i;
    cout << "data[" << i << "] = " << i << endl;
  }

  // Example memory leak
  //delete x;

  delete[] data;

  cout << "All done!" << endl;

  return 0;
}
