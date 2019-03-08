"""
Programa desenvolvido com a inteção de prever os balores das cotações da Bolsa de Valores
Os arquivos utilizados com os históricos 
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/


"""
import Bovespatofile as Rd
import Indicadores as Ind

# Descriptografia dos dados baixados da B3

filepath  = 'D:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos'
filename  = 'COTACAO_'
stockname = ['ABCP11','BOVA11','ITUB4','VALE4','PETR4','TIET11','SANB11','ELET6','ABEV3','GOAU4','CMIG4']
year      = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

Rd.readFromBovespa(filepath,filename,stockname,year)



