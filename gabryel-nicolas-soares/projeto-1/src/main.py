import requests

memoria = {}

# CHAMADA LLM (OLLAMA)
def chamar_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]
    except:
        return "Erro ao acessar o modelo."


# CLASSIFICAÇÃO DE INTENÇÃO
def classificar_intencao(mensagem):
    mensagem = mensagem.lower()

    if "beneficio" in mensagem or "direito" in mensagem:
        return "triagem"
    elif mensagem in ["sair", "exit"]:
        return "encerramento"
    else:
        return "pergunta"

# COLETA DE DADOS
def coletar_dados():
    print("\n🔎 Vamos analisar seu direito a benefícios:\n")

    renda = float(input("Renda familiar mensal (R$): "))
    pessoas = int(input("Número de pessoas na casa: "))
    idade = int(input("Sua idade: "))

    renda_per_capita = renda / pessoas

    dados = {
        "renda": renda,
        "pessoas": pessoas,
        "idade": idade,
        "renda_per_capita": renda_per_capita
    }

    return dados

# MOTOR DE ELEGIBILIDADE
def motor_elegibilidade(dados):
    beneficios = []

    if dados["renda_per_capita"] <= 218:
        beneficios.append("Bolsa Família")

    if dados["renda_per_capita"] <= 353 and dados["idade"] >= 65:
        beneficios.append("BPC (Benefício de Prestação Continuada)")

    if not beneficios:
        resposta = f"""
❌ Você NÃO tem direito aos principais benefícios analisados.

Motivo:
- Sua renda por pessoa é R$ {dados['renda_per_capita']:.2f}, acima dos limites.

Próximos passos:
- Procure o CRAS da sua cidade para uma avaliação completa.
"""
        memoria["ultima_resposta"] = resposta
        return resposta

    prompt = f"""
    Explique de forma direta.

    Benefícios aprovados:
    {beneficios}

    Dados:
    {dados}

    Regras:
    - NÃO inventar benefícios
    - NÃO fazer perguntas
    - NÃO sair do tema
    - Seja objetivo

    Formato:

    Benefícios:
    - ...

    Motivo:
    - ...

    Próximos passos:
    - ...
    """

    resposta = chamar_llm(prompt)
    memoria["ultima_resposta"] = resposta

    return resposta

# PERGUNTAS
def responder_pergunta(pergunta):
    ultima = memoria.get("ultima_resposta", "")

    prompt = f"""
    Responda com base nisso:

    {ultima}

    Pergunta:
    {pergunta}

    Regras:
    - Responder direto
    - Não mudar de assunto
    """

    return chamar_llm(prompt)

# AGENTE PRINCIPAL
def agente():
    print("🤖 Agente de Assistência Social")
    print("Objetivo: verificar direito a benefícios do governo")
    print("Digite 'sair' para encerrar.\n")

    while True:
        msg = input("Você: ")

        if msg.lower() in ["sair", "exit"]:
            print("Encerrando atendimento...")
            break

        intencao = classificar_intencao(msg)

        if intencao == "triagem":
            dados = coletar_dados()
            memoria["dados_usuario"] = dados

            resposta = motor_elegibilidade(dados)
            print("\n📋 Resultado:\n")
            print(resposta)

        elif intencao == "pergunta":
            resposta = responder_pergunta(msg)
            print("\n💬", resposta)

        else:
            print("\n🤖 Posso verificar seu direito a benefícios.")

# EXECUÇÃO
if __name__ == "__main__":
    agente()