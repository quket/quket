pqrstuvw = ['p','q','r','s','t','u','v','w']
with open('func_spinorbital.py') as f:
    texts = f.readlines()
HD_list = []
with open('_func_spinorbital.py', 'w') as f:
    for line in texts:
        space = 0
        for k in line:
            if k == ' ':
                space += 1
                print(' ', file=f, end='')
            else:
                break
        line_ = line.split()
        for j, word in enumerate(line_):
            end_parentheses = line_[-1] == '))'
            end_parenthesis = line_[-1] == ')'
            if word == '+einsum(' or word == '-einsum(':
                # Change lines like 
                #   +einsum( 'ij,ij->', HD['Habba'][p,:,t,:], HD['Dbaababba'][u,r,s,:,q,v,w,:] ) ,
                # to
                #   +HD['Habba13Dbaababba37'][p,t,u,r,s,q,v,w]
                # and obtain new definitions for HD,
                #    'Habba13Dbaababba37' : einsum('piqj,rstiuvwj->pqrstuvw', HD['Habba'], HD['Dbaabaaba']), 
                contraction = line_[j+1]
                Hamiltonian = line_[j+2]
                RDM = line_[j+3]

                 
                ### contraction_list is ['i', 'j']
                contraction_list = contraction.split(",")[0].replace("'","")
                ### Hspin is "Habba"
                Hspin = Hamiltonian.split("'")[1]
                ### Hblock is ['p', ':', 't', ':']
                Hblock = Hamiltonian.replace("]","").split("[")[2].split(',')
                try:
                    Hblock.remove('')
                except:
                    pass
                # Number of labels
                Nlabel = len(Hblock) - Hblock.count(':')
                ### Dspin is "Dbaababba"
                Dspin = RDM.split("'")[1]
                ### Dblock is ['u','r','s',':','q','v','w',':']
                Dblock = RDM.replace("]","").split("[")[2].split(',')
                try:
                    Dblock.remove('')
                except:
                    pass
                # Number of labels
                Nlabel += len(Dblock) - Dblock.count(':')

                ### Set new HD term
                contraction_H = ""
                contraction_D = ""
                H_position = ""    ## position of : in H contraction
                D_position = ""    ## position of : in D contraction
                index = ""         ## index of new HD
                k = 0
                l = 0
                for i in range(len(Hblock)):
                    if Hblock[i] == ":":
                        contraction_H += contraction_list[k]
                        H_position += str(i)
                        k +=1
                    else:
                        contraction_H += pqrstuvw[l]
                        index += Hblock[i] + ","
                        l += 1
                k = 0
                for i in range(len(Dblock)):
                    if Dblock[i] == ":":
                        contraction_D += contraction_list[k]
                        D_position += str(i)
                        k +=1
                    else:
                        contraction_D += pqrstuvw[l]
                        index += Dblock[i] + ","
                        l += 1
                
                index = index[:-1]
                contraction_new = contraction_H + ',' + contraction_D + '->'
                for i in range(Nlabel):
                    contraction_new += pqrstuvw[i]
                ### contraction_new is 'piqj,rstiuvwj->pqrstuvw'
                #print(contraction_new)

                ### key is "'Habba13Dbaaababba37'"
                key = "'" + Hspin + H_position + Dspin + D_position + "'"
                #print(key)
                ### Construct new line
                #   +HD['Habba13Dbaababba37'][p,t,u,r,s,q,v,w]
                sign = word[0]
                new_line = f"{sign}HD[{key}][{index}]"
                if end_parentheses:
                    new_line += " ))"
                elif end_parenthesis:
                    new_line += ""
                else:
                    new_line += " ,"
                #print(new_line)
                print(new_line, file=f)
                #  HD_ is   'Habba13Dbaababba37' : einsum('piqj,rstiuvwj->pqrstuvw', HD['Habba'], HD['Dbaabaaba']), 
                HD_ = f"{key} : einsum('{contraction_new}', HD['{Hspin}'], HD['{Dspin}'])," 
                if HD_ not in HD_list:
                    HD_list.append(HD_)
                break
            elif j == len(line_)-1:
                print(word,file=f, end='')
            else:
                print(word+' ',file=f, end='')
        else:
            print('',file=f)


    print("ADD THE FOLLOWING TO DEFINE ADDITIONAL HD:")
    print("------------------------------------------")
    print('    HD.update({')
    for HD_ in HD_list:
        print('        '+HD_)
    print('    })')
