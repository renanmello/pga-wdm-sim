import socket
HOST = '127.0.0.1'     # Endereco IP do Servidor
PORT = 5000            # Porta que o Servidor esta
tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
dest = (HOST, PORT)
tcp.connect(dest)

msg = input('type number of simulations and cores example: 2;2\n')

while msg != '\x18':
    tcp.send(msg.encode())
    file = tcp.recv(1024)
    print('\n simulations times:\n')
    print(file.decode('UTF-8'))
    print('\n Bloking probability:\n')
    file2 = tcp.recv(1024)
    print(file2.decode('UTF-8'))
    msg = input()
tcp.close()