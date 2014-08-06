
#include <iostream>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// needed for sending/receive mpi
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

// ---------------------------------------

class Cat
{
      private:
        int name_ ;
        std::vector<double> hairList_ ;

      private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & name_ ;
            ar & hairList_ ;
        }

      public:
        Cat(){}
        Cat(const int & newName, 
            const std::vector<double> & newHairList)
          : name_(newName), hairList_(newHairList)
        {}
        ~Cat(){}

        int getName()
        { return name_ ; }
};



int main(int argc, char *argv[])
{
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  boost::shared_ptr<Cat> initCat ;
  boost::shared_ptr<Cat> recvdCat ;

  int NUM_INFO_EXCHANGES = 100 ;
  if ( world.rank()==0 )
  {
      std::vector<double> initHairList (5000, 1.5);
      initCat = boost::make_shared<Cat>( 111, initHairList) ;
      world.send(1, 0, initCat);        

      for(int i = 0 ; i < NUM_INFO_EXCHANGES ; i++)
      {
        std::cout << " Iteration = " << i << " of " << NUM_INFO_EXCHANGES << std::endl;
        world.recv< boost::shared_ptr<Cat> >(1, 1, recvdCat);
        world.send< boost::shared_ptr<Cat> >(1, 0, initCat);        
      }

      world.recv< boost::shared_ptr<Cat> >(1, 1, recvdCat);
  }
  else
  {
      for(int i = 0 ; i < NUM_INFO_EXCHANGES+1 ; i++)
      {
        world.recv(0, 0, recvdCat);
        assert( recvdCat->getName() == 111 );
        world.send< boost::shared_ptr<Cat> >(0, 1, recvdCat);        
      }
  }

  return 0;
}

