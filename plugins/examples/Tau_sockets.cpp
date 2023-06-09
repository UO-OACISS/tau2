
#include "Tau_sockets.h"
#include "Tau_scoped_timer.h"
#include <iostream>

#include <arpa/inet.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/param.h>

#include <fstream>
#include <time.h>
#include <unordered_map>
#ifdef TAU_MPI
#include "mpi.h"
#endif

static std::unordered_map<int, tau::plugins::HostInfo> hosts;

// Server side C/C++ program to demonstrate Socket programming
void tau::plugins::Sockets::Run(int rank, tau::plugins::CallbackFunctionType * cb)
{
    tau::plugins::ScopedTimer(__func__);
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                                                &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    //printf("%d Binding to port %d\n", rank, hosts[rank]->_port);fflush(stdout);
    address.sin_port = htons( hosts[rank]._port );

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                        (socklen_t*)&addrlen))<0)
        {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        valread = read( new_socket , buffer, 1024);
        if (valread > 0) {
            //printf("%s\n",buffer ); fflush(stdout);
            // null message means exit.
            tau::plugins::ScopedTimer("Processing Remote Request");
            if (strlen(buffer) == 0) {
                const char * reply = "Exiting.";
                send(new_socket , reply , strlen(reply) , 0 );
                break;
            } else {
                char * reply = (*cb)(buffer);
                send(new_socket , reply , strlen(reply) , 0 );
                free(reply);
            }
        }
        close (new_socket);
    }
    close(server_fd);
    return;
}


// Client side C/C++ program to demonstrate Socket programming
char * tau::plugins::Sockets::send_message(int rank, const char * message)
{
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n"); fflush(stdout);
        return nullptr;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(hosts[rank]._port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)
    {
        printf("\nInvalid address/ Address not supported \n"); fflush(stdout);
        return nullptr;
    }

    int max_failures = 100;
    while (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
        max_failures--;
        perror("failed to connect. ");
        printf("...%d attempts left\n", max_failures);

        if (max_failures == 0) {
            printf("\nConnection Failed \n"); fflush(stdout);
            return nullptr;
        }
    }
    send(sock , message , strlen(message) , 0 );
    valread = read( sock , buffer, 1024);
    if (valread > 0) {
        //printf("%s\n",buffer ); fflush(stdout);
    }
    close(sock);
    return strdup(buffer);
}

void tau::plugins::Sockets::GetHostInfo(int port) {
#ifdef TAU_MPI
    // figure out who should get system stats for this node
    int i;
    int my_rank = RtsLayer::myNode();
    int comm_size = tau_totalnodes(0,1);

    // get my hostname
    const int hostlength = 128;
    char hostname[hostlength] = {0};
    gethostname(hostname, sizeof(char)*hostlength);
    // get my IP address
    hostent * record = gethostbyname(hostname);
    if(record == NULL) {
        perror("host is unavailable?\n");
        exit(1);
    }
    in_addr * address = (in_addr * )record->h_addr;
    const int addrlength = 32;
    char ip_address[addrlength] = {0};
    sprintf(ip_address, "%s", inet_ntoa(* address));

    // make array for all hostnames
    char * allhostnames = (char*)calloc(hostlength * comm_size, sizeof(char));
    // make array for all addresses
    char * alladdresses = (char*)calloc(addrlength * comm_size, sizeof(char));

    // copy my name into the big array
    char * host_index = allhostnames + (hostlength * my_rank);
    strncpy(host_index, hostname, hostlength);
    // copy my address into the big array
    char * addr_index = alladdresses + (addrlength * my_rank);
    strncpy(addr_index, ip_address, addrlength);

    // get all hostnames
    PMPI_Allgather(hostname, hostlength, MPI_CHAR, allhostnames,
                   hostlength, MPI_CHAR, MPI_COMM_WORLD);
    // get all addresses
    PMPI_Allgather(ip_address, addrlength, MPI_CHAR, alladdresses,
                   addrlength, MPI_CHAR, MPI_COMM_WORLD);

    // point to the head of the array
    host_index = allhostnames;
    addr_index = alladdresses;

    // find the hostname, addr, and port for all ranks
    char * previous_host = host_index;
    int port_index = -1; // so that the first one will be 0
    for (i = 0 ; i < comm_size ; i++) {
        //printf("%d:%d comparing '%s' to '%s'\n", my_rank, comm_size, previous_host, host_index);fflush(stdout);
        if (strncmp(previous_host, host_index, hostlength) == 0) {
            // these are the same
            port_index++;
        } else {
            port_index = 0;
        }
        //tau::plugins::HostInfo * info =
        //    new tau::plugins::HostInfo(host_index, addr_index, port_index + port);
        tau::plugins::HostInfo info(host_index, addr_index, port_index + port);
        hosts.insert(std::make_pair(i, info));
        // advance to next
        previous_host = host_index;
        host_index = host_index + hostlength;
        addr_index = addr_index + addrlength;
    }
    free(allhostnames);
    free(alladdresses);
    return;
#else
    return;
#endif
}

