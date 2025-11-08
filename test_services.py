"""
Quick Service Connection Test
Tests connectivity to all infrastructure services
"""

import sys
import psycopg2
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient
import requests


def test_spark():
    """Test Spark Master UI accessibility"""
    print("\n🔥 Testing Spark Master...")
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✅ Spark Master UI is accessible at http://localhost:8080")
            return True
        else:
            print(f"❌ Spark Master returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Spark Master connection failed: {e}")
        return False


def test_postgresql():
    """Test PostgreSQL database connection"""
    print("\n🐘 Testing PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="campaign_analytics",
            user="postgres",
            password="password",
            connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"✅ PostgreSQL connected successfully")
        print(f"   Version: {version[:50]}...")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


def test_kafka():
    """Test Kafka broker connection"""
    print("\n📨 Testing Kafka...")
    try:
        # Test with admin client first
        admin_client = KafkaAdminClient(
            bootstrap_servers='localhost:9092',
            request_timeout_ms=5000
        )
        topics = admin_client.list_topics()
        print(f"✅ Kafka broker is accessible at localhost:9092")
        print(f"   Available topics: {list(topics)}")
        admin_client.close()
        
        # Test producer
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            request_timeout_ms=5000
        )
        print("✅ Kafka Producer created successfully")
        producer.close()
        
        return True
    except Exception as e:
        print(f"❌ Kafka connection failed: {e}")
        return False


def test_flask_api():
    """Test Flask API (if running)"""
    print("\n🌐 Testing Flask API...")
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ Flask API is running at http://localhost:5000")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Flask API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ℹ️  Flask API is not running (start with: python api\\app.py)")
        return None
    except Exception as e:
        print(f"❌ Flask API test failed: {e}")
        return False


def test_streamlit():
    """Test Streamlit Dashboard (if running)"""
    print("\n📊 Testing Streamlit Dashboard...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit Dashboard is running at http://localhost:8501")
            return True
        else:
            print(f"❌ Streamlit returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ℹ️  Streamlit Dashboard is not running (start with: streamlit run dashboard\\app.py)")
        return None
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False


def main():
    """Run all connectivity tests"""
    print("=" * 60)
    print("🔍 RL CAMPAIGN OPTIMIZER - SERVICE CONNECTIVITY TEST")
    print("=" * 60)
    
    results = {}
    
    # Test core infrastructure services
    results['Spark'] = test_spark()
    results['PostgreSQL'] = test_postgresql()
    results['Kafka'] = test_kafka()
    
    # Test application services (optional)
    results['Flask API'] = test_flask_api()
    results['Streamlit'] = test_streamlit()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for service, status in results.items():
        if status is True:
            print(f"✅ {service:20s} - CONNECTED")
        elif status is False:
            print(f"❌ {service:20s} - FAILED")
        else:
            print(f"ℹ️  {service:20s} - NOT RUNNING")
    
    print("\n" + "=" * 60)
    
    # Check if core services are all running
    core_services = ['Spark', 'PostgreSQL', 'Kafka']
    core_status = [results[s] for s in core_services]
    
    if all(core_status):
        print("🎉 All core infrastructure services are running!")
        print("\n📚 Next Steps:")
        print("   1. Start Flask API:    python api\\app.py")
        print("   2. Start Dashboard:    streamlit run dashboard\\app.py")
        print("   3. Run Kafka Producer: python src\\kafka_producer.py")
        print("\n📖 See SERVICE_ACCESS_GUIDE.md for detailed access instructions")
        return 0
    else:
        print("⚠️  Some core services failed. Check docker-compose logs:")
        print("   docker-compose logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
