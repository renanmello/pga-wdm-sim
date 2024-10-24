import socket
import sim as simulator
import os , sys

HOST = ''              # Endereco IP do Servidor
PORT = 5000            # Porta que o Servidor esta
tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
orig = (HOST, PORT)
tcp.bind(orig)
tcp.listen(1)
filename = 'E:\\tccproject\\time.txt'
filename2 = 'E:\\tccproject\\simulacoes.txt'



while True:
    con, cliente = tcp.accept()
    while True:
        msg = con.recv(1024)
        translate = msg.decode('UTF-8')
        split = translate.split(';')
        print(split[1], split[0])
        os.system('mpiexec -n ' + split[1] + ' python sim.py ' + split[0])
        file = open(filename, "r", encoding="utf-8")
        skar = file.read(20000)

        file2 = open(filename2, "r", encoding="utf-8")
        skar2 = file2.read(20000)

        con.send(skar.encode())
        con.send(skar2.encode())

        if not msg: break
    con.close()
