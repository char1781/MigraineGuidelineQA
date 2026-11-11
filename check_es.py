from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])
index_name = 'medical_guidelines'

print(f"正在连接 ES: {es.info().get('version', {}).get('number')} ...")

if es.indices.exists(index=index_name):
    # 1. 查总数
    count = es.count(index=index_name)['count']
    print(f"✅ 当前 ES 中的文档总数: {count} 条")
    
    # 2. 查前 3 条看看内容
    if count > 0:
        res = es.search(index=index_name, body={'size': 3})
        print("\n前3条数据示例:")
        for hit in res['hits']['hits']:
            src = hit['_source']
            guide = src.get('source', {}).get('guideline', '未知')
            drug = src.get('intervention', {}).get('name', '未知')
            print(f"- 指南: {guide} | 药物: {drug}")
else:
    print(f"⚠️ ES 中不存在索引 '{index_name}'，说明目前没有任何数据。")