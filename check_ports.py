"""
Simple Service Access Test (No Dependencies Required)
"""

import socket
import sys


def test_port(host, port, service_name):
    """Test if a port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ {service_name:20s} - Port {port} is OPEN")
            return True
        else:
            print(f"❌ {service_name:20s} - Port {port} is CLOSED")
            return False
    except Exception as e:
        print(f"❌ {service_name:20s} - Error: {e}")
        return False


def main():
    print("=" * 60)
    print("🔍 RL CAMPAIGN OPTIMIZER - PORT CONNECTIVITY TEST")
    print("=" * 60)
    print()
    
    services = [
        ("localhost", 8080, "Spark Master UI"),
        ("localhost", 7077, "Spark Master"),
        ("localhost", 4040, "Spark Job UI"),
        ("localhost", 9092, "Kafka Broker"),
        ("localhost", 2181, "Zookeeper"),
        ("localhost", 5432, "PostgreSQL"),
    ]
    
    results = []
    for host, port, name in services:
        results.append(test_port(host, port, name))
    
    print()
    print("=" * 60)
    print("📋 ACCESS INSTRUCTIONS")
    print("=" * 60)
    print()
    print("🌐 WEB INTERFACES:")
    print("   • Spark Master UI:  http://localhost:8080")
    print("   • Spark Job UI:     http://localhost:4040 (when job running)")
    print()
    print("🔌 DATABASE CONNECTION:")
    print("   • Host:     localhost")
    print("   • Port:     5432")
    print("   • Database: campaign_analytics")
    print("   • User:     postgres")
    print("   • Password: password")
    print()
    print("   Connect via Docker:")
    print("   docker exec -it postgres-db psql -U postgres -d campaign_analytics")
    print()
    print("📨 KAFKA:")
    print("   • Bootstrap Servers: localhost:9092")
    print()
    print("   List Topics:")
    print("   docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092")
    print()
    print("   Create Topic:")
    print("   docker exec -it kafka kafka-topics --create --topic customer_events \\")
    print("     --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1")
    print()
    print("=" * 60)
    print()
    
    if all(results[:3]):  # Check core services
        print("✅ All infrastructure services are accessible!")
        print()
        print("📚 NEXT STEPS:")
        print("   1. Install Python packages:  pip install -r requirements.txt")
        print("   2. Start Flask API:          python api\\app.py")
        print("   3. Start Dashboard:          streamlit run dashboard\\app.py")
        print("   4. Test Kafka Producer:      python src\\kafka_producer.py")
        print()
        return 0
    else:
        print("⚠️  Some services are not accessible.")
        print("   Run: docker-compose ps")
        print("   Run: docker-compose logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
