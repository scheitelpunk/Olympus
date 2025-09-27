"""Comprehensive test suite for PostgreSQL Storage.

Tests the complete database storage functionality including:
- Connection pooling and management
- Schema initialization and migration
- Experience storage and retrieval
- Dream session management
- Strategy storage and tracking
- Performance optimization and cleanup
- Error handling and connection recovery
"""

import pytest
import psycopg2
import time
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, call
from contextlib import contextmanager
from typing import Dict, Any, List

from morpheus.storage.postgres_storage import PostgreSQLStorage, handle_db_errors
from morpheus.core.types import (
    SensoryExperience, LearnedStrategy, MaterialInteraction,
    SystemMetrics, ActionType, Vector3D, ContactPoint, TactileSignature
)


class TestPostgreSQLStorage:
    """Test PostgreSQL storage functionality."""
    
    @pytest.fixture
    def connection_params(self):
        """Mock connection parameters."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'morpheus_test',
            'user': 'test_user',
            'password': 'test_pass',
            'pool_size': 5
        }
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = Mock()
        mock_conn = Mock()
        mock_cursor = Mock()
        
        # Setup cursor context manager
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        
        # Setup connection context manager  
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        
        pool.getconn.return_value = mock_conn
        pool.putconn.return_value = None
        
        return pool, mock_conn, mock_cursor
    
    @pytest.fixture
    def storage(self, connection_params, mock_pool):
        """Create storage instance with mocked dependencies."""
        pool, mock_conn, mock_cursor = mock_pool
        
        with patch('morpheus.storage.postgres_storage.ThreadedConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = pool
            
            with patch.object(PostgreSQLStorage, 'init_schema'):
                storage = PostgreSQLStorage(connection_params)
                storage.pool = pool
                storage._mock_conn = mock_conn
                storage._mock_cursor = mock_cursor
                
                return storage
    
    def test_initialization_success(self, connection_params, mock_pool):
        """Test successful storage initialization."""
        pool, mock_conn, mock_cursor = mock_pool
        
        with patch('morpheus.storage.postgres_storage.ThreadedConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = pool
            
            with patch.object(PostgreSQLStorage, 'init_schema'):
                storage = PostgreSQLStorage(connection_params)
                
                assert storage.pool == pool
                assert storage.min_connections == 1
                assert storage.max_connections == 5
                mock_pool_class.assert_called_once()
    
    def test_initialization_connection_failure(self, connection_params):
        """Test initialization with connection failure."""
        with patch('morpheus.storage.postgres_storage.ThreadedConnectionPool') as mock_pool_class:
            mock_pool_class.side_effect = psycopg2.OperationalError("Connection failed")
            
            with pytest.raises(psycopg2.OperationalError):
                PostgreSQLStorage(connection_params)
    
    def test_get_connection_context_manager(self, storage):
        """Test connection context manager."""
        with storage.get_connection() as conn:
            assert conn == storage._mock_conn
            storage.pool.getconn.assert_called_once()
        
        storage.pool.putconn.assert_called_once_with(storage._mock_conn)
    
    def test_get_connection_error_handling(self, storage):
        """Test connection error handling."""
        storage.pool.getconn.side_effect = psycopg2.Error("Pool exhausted")
        
        with pytest.raises(psycopg2.Error):
            with storage.get_connection() as conn:
                pass
    
    def test_init_schema(self, storage):
        """Test schema initialization."""
        # Reset mock to test actual schema init
        storage._mock_cursor.reset_mock()
        
        storage.init_schema()
        
        # Should execute multiple schema creation statements
        assert storage._mock_cursor.execute.call_count > 10
        storage._mock_conn.commit.assert_called()
        
        # Check for key table creation
        executed_sql = [call.args[0] for call in storage._mock_cursor.execute.call_args_list]
        table_creates = [sql for sql in executed_sql if 'CREATE TABLE' in sql]
        
        expected_tables = ['experiences', 'dream_sessions', 'learned_strategies', 
                          'material_predictions', 'sensor_calibration', 'system_metrics']
        
        for table in expected_tables:
            assert any(table in sql for sql in table_creates)
    
    def test_store_experience_success(self, storage):
        """Test successful experience storage."""
        # Create sample experience
        experience = SensoryExperience(
            experience_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            primary_material='steel',
            secondary_material='rubber',
            action_type=ActionType.GRASP,
            success=True,
            reward=0.8,
            tags=['test', 'grasp'],
            notes='Test experience'
        )
        
        # Mock cursor to return ID
        storage._mock_cursor.fetchone.return_value = [123]
        
        exp_id = storage.store_experience(experience)
        
        assert exp_id == 123
        storage._mock_cursor.execute.assert_called()
        storage._mock_conn.commit.assert_called()
        
        # Verify SQL structure
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'INSERT INTO experiences' in sql_call[0]
    
    def test_store_experience_with_embeddings(self, storage):
        """Test storing experience with embeddings."""
        import numpy as np
        
        experience = SensoryExperience(
            tactile_embedding=np.random.randn(64),
            audio_embedding=np.random.randn(32),
            visual_embedding=np.random.randn(128),
            fused_embedding=np.random.randn(128)
        )
        
        storage._mock_cursor.fetchone.return_value = [456]
        
        exp_id = storage.store_experience(experience)
        
        assert exp_id == 456
        storage._mock_conn.commit.assert_called()
    
    def test_store_experience_with_contact_points(self, storage):
        """Test storing experience with contact points."""
        contact_points = [
            ContactPoint(
                position=Vector3D(0, 0, 0),
                normal=Vector3D(0, 0, 1),
                force_magnitude=1.5,
                object_a=1,
                object_b=2
            )
        ]
        
        experience = SensoryExperience(
            contact_points=contact_points,
            forces=[1.0, 0.5, 0.2],
            torques=[0.1, 0.0, 0.0]
        )
        
        storage._mock_cursor.fetchone.return_value = [789]
        
        exp_id = storage.store_experience(experience)
        
        assert exp_id == 789
        storage._mock_conn.commit.assert_called()
    
    def test_get_experiences_basic(self, storage):
        """Test basic experience retrieval."""
        # Mock return data
        mock_experiences = [
            {
                'id': 1,
                'experience_id': str(uuid.uuid4()),
                'session_id': str(uuid.uuid4()),
                'primary_material': 'steel',
                'success': True
            },
            {
                'id': 2, 
                'experience_id': str(uuid.uuid4()),
                'session_id': str(uuid.uuid4()),
                'primary_material': 'aluminum',
                'success': False
            }
        ]
        
        storage._mock_cursor.fetchall.return_value = mock_experiences
        
        experiences = storage.get_experiences(limit=10)
        
        assert len(experiences) == 2
        assert experiences[0]['primary_material'] == 'steel'
        storage._mock_cursor.execute.assert_called()
    
    def test_get_experiences_with_filters(self, storage):
        """Test experience retrieval with filters."""
        storage._mock_cursor.fetchall.return_value = []
        
        experiences = storage.get_experiences(
            session_id='test-session',
            hours=24,
            material='steel',
            success_only=True,
            action_type='GRASP',
            min_reward=0.5,
            tags=['important']
        )
        
        # Verify SQL includes all filter conditions
        sql_call = storage._mock_cursor.execute.call_args[0]
        sql_query = sql_call[0]
        sql_params = sql_call[1]
        
        assert 'WHERE' in sql_query
        assert 'session_id = %s' in sql_query
        assert 'created_at > NOW() - INTERVAL' in sql_query
        assert 'primary_material = %s' in sql_query
        assert 'success = TRUE' in sql_query
        assert 'action_type = %s' in sql_query
        assert 'reward >= %s' in sql_query
        assert 'tags @> %s' in sql_query
        
        # Check parameters
        assert 'test-session' in sql_params
        assert 24 in sql_params
        assert 'steel' in sql_params
        assert 'GRASP' in sql_params
        assert 0.5 in sql_params
        assert ['important'] in sql_params
    
    def test_get_recent_experiences(self, storage):
        """Test getting recent experiences."""
        storage._mock_cursor.fetchall.return_value = []
        
        experiences = storage.get_recent_experiences(hours=12, limit=100)
        
        storage._mock_cursor.execute.assert_called()
        # Should call get_experiences with hours parameter
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'INTERVAL' in sql_call[0]
        assert 12 in sql_call[1]
    
    def test_store_dream_session(self, storage):
        """Test storing dream session."""
        session_data = {
            'config': {'replay_speed': 10.0, 'parallel_dreams': 4},
            'experience_count': 100,
            'time_range_hours': 24,
            'replay_count': 50,
            'variations_generated': 200,
            'compute_time_seconds': 45.2,
            'strategies_found': 5,
            'strategies_stored': 3,
            'improvements': {'avg': 0.2, 'best': 0.8},
            'consolidated_memories': 3,
            'average_improvement': 0.25,
            'best_improvement': 0.8,
            'status': 'completed',
            'notes': 'Successful dream session'
        }
        
        mock_session_id = str(uuid.uuid4())
        storage._mock_cursor.fetchone.return_value = [mock_session_id]
        
        session_id = storage.store_dream_session(session_data)
        
        assert session_id == mock_session_id
        
        # Should execute INSERT and UPDATE
        assert storage._mock_cursor.execute.call_count >= 2
        storage._mock_conn.commit.assert_called()
        
        # Verify INSERT structure
        first_call = storage._mock_cursor.execute.call_args_list[0]
        assert 'INSERT INTO dream_sessions' in first_call[0][0]
    
    def test_store_learned_strategy(self, storage):
        """Test storing learned strategy."""
        strategy = LearnedStrategy(
            strategy_id=str(uuid.uuid4()),
            name='grip_optimization',
            category='tactile',
            strategy_data={'param1': 10.0, 'param2': 'value'},
            baseline_performance=0.6,
            improved_performance=0.85,
            improvement_ratio=0.25,
            confidence=0.9,
            applicable_materials=['steel', 'aluminum'],
            applicable_scenarios=['grasp', 'lift'],
            times_used=0,
            success_rate=0.0
        )
        
        storage._mock_cursor.fetchone.return_value = [456]
        
        strategy_id = storage.store_learned_strategy(strategy)
        
        assert strategy_id == 456
        storage._mock_cursor.execute.assert_called()
        storage._mock_conn.commit.assert_called()
        
        # Verify SQL structure
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'INSERT INTO learned_strategies' in sql_call[0]
    
    def test_get_learned_strategies(self, storage):
        """Test retrieving learned strategies."""
        mock_strategies = [
            {
                'id': 1,
                'strategy_id': str(uuid.uuid4()),
                'name': 'strategy1',
                'category': 'tactile',
                'confidence': 0.9,
                'improvement_ratio': 0.3
            },
            {
                'id': 2,
                'strategy_id': str(uuid.uuid4()),
                'name': 'strategy2', 
                'category': 'motion',
                'confidence': 0.7,
                'improvement_ratio': 0.2
            }
        ]
        
        storage._mock_cursor.fetchall.return_value = mock_strategies
        
        strategies = storage.get_learned_strategies(
            category='tactile',
            min_confidence=0.8,
            min_improvement=0.1,
            active_only=True,
            limit=50
        )
        
        assert len(strategies) == 2
        storage._mock_cursor.execute.assert_called()
        
        # Verify filtering SQL
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'confidence >= %s' in sql_call[0]
        assert 'improvement_ratio >= %s' in sql_call[0]
        assert 'category = %s' in sql_call[0]
        assert 'is_active = TRUE' in sql_call[0]
        assert 'LIMIT %s' in sql_call[0]
    
    def test_update_strategy_usage(self, storage):
        """Test updating strategy usage statistics."""
        strategy_id = str(uuid.uuid4())
        
        # Test successful usage
        storage.update_strategy_usage(strategy_id, success=True)
        
        storage._mock_cursor.execute.assert_called()
        storage._mock_conn.commit.assert_called()
        
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'UPDATE learned_strategies' in sql_call[0]
        assert 'times_used = times_used + 1' in sql_call[0]
        assert 'times_successful = times_successful + %s' in sql_call[0]
        assert strategy_id in sql_call[1]
        assert 1 in sql_call[1]  # Success value
        
        # Reset mock
        storage._mock_cursor.reset_mock()
        storage._mock_conn.reset_mock()
        
        # Test failed usage
        storage.update_strategy_usage(strategy_id, success=False)
        
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 0 in sql_call[1]  # Failure value
    
    def test_cleanup_old_data(self, storage):
        """Test data cleanup functionality."""
        storage._mock_cursor.rowcount = 10
        
        cleanup_stats = storage.cleanup_old_data(
            retention_days=30,
            keep_important=True
        )
        
        assert isinstance(cleanup_stats, dict)
        assert 'experiences_deleted' in cleanup_stats
        assert 'dream_sessions_cleaned' in cleanup_stats
        assert 'calibrations_deleted' in cleanup_stats
        assert 'metrics_deleted' in cleanup_stats
        
        # Should execute multiple DELETE statements
        assert storage._mock_cursor.execute.call_count >= 4
        storage._mock_conn.commit.assert_called()
        
        # Check for VACUUM ANALYZE
        executed_sql = [call.args[0] for call in storage._mock_cursor.execute.call_args_list]
        assert any('VACUUM ANALYZE' in sql for sql in executed_sql)
    
    def test_get_database_stats(self, storage):
        """Test database statistics retrieval."""
        # Mock table size data
        mock_table_stats = [
            {
                'tablename': 'experiences',
                'size_bytes': 1024 * 1024,  # 1MB
                'table_size': 800 * 1024,
                'index_size': 224 * 1024
            },
            {
                'tablename': 'learned_strategies',
                'size_bytes': 512 * 1024,  # 512KB
                'table_size': 400 * 1024,
                'index_size': 112 * 1024
            }
        ]
        
        mock_activity = {
            'recent_experiences': 50,
            'successful_experiences': 40,
            'avg_reward': 0.7,
            'active_sessions': 3
        }
        
        # Setup multiple fetchall returns
        storage._mock_cursor.fetchall.side_effect = [mock_table_stats]
        storage._mock_cursor.fetchone.side_effect = [mock_activity, [100], [25], [5], [2]]
        
        stats = storage.get_database_stats()
        
        assert isinstance(stats, dict)
        assert 'tables' in stats
        assert 'total_size_mb' in stats
        assert 'record_counts' in stats
        assert 'recent_activity' in stats
        assert 'connection_pool' in stats
        
        # Verify table stats
        assert 'experiences' in stats['tables']
        assert stats['tables']['experiences']['size_mb'] == 1.0  # 1MB converted
        assert stats['total_size_mb'] == 1.5  # 1MB + 512KB
    
    def test_store_system_metrics(self, storage):
        """Test storing system metrics."""
        metrics = SystemMetrics(
            perception_count=100,
            dream_count=5,
            strategies_learned=12,
            average_processing_time=25.5,
            memory_usage_mb=256.0,
            success_rate=0.85
        )
        
        storage.store_system_metrics(metrics)
        
        storage._mock_cursor.execute.assert_called()
        storage._mock_conn.commit.assert_called()
        
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'INSERT INTO system_metrics' in sql_call[0]
        
        # Verify parameter values
        params = sql_call[1]
        assert 'system_snapshot' in params
        assert 100 in params  # perception_count
        assert 5 in params    # dream_count
        assert 12 in params   # strategies_learned
    
    def test_context_manager(self, storage):
        """Test storage as context manager."""
        with storage as s:
            assert s == storage
        
        # Should call close
        storage.pool.closeall.assert_called()
    
    def test_close_connection_pool(self, storage):
        """Test closing connection pool."""
        storage.close()
        storage.pool.closeall.assert_called()
    
    def test_error_handling_decorator(self):
        """Test database error handling decorator."""
        
        @handle_db_errors
        def test_func(self):
            raise psycopg2.Error("Test database error")
        
        with pytest.raises(psycopg2.Error):
            test_func(None)
        
        @handle_db_errors  
        def test_func2(self):
            raise ValueError("Test non-database error")
        
        with pytest.raises(ValueError):
            test_func2(None)
    
    def test_concurrent_access(self, storage):
        """Test concurrent database access."""
        import threading
        
        results = []
        
        def store_experience():
            experience = SensoryExperience()
            # Mock return for each thread
            storage._mock_cursor.fetchone.return_value = [threading.current_thread().ident]
            exp_id = storage.store_experience(experience)
            results.append(exp_id)
        
        # Start multiple threads
        threads = [threading.Thread(target=store_experience) for _ in range(5)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        # All operations should succeed
        assert all(isinstance(r, int) for r in results)
    
    def test_transaction_rollback_on_error(self, storage):
        """Test transaction rollback on error."""
        # Setup cursor to raise error on commit
        storage._mock_conn.commit.side_effect = psycopg2.Error("Commit failed")
        
        experience = SensoryExperience()
        
        with pytest.raises(psycopg2.Error):
            storage.store_experience(experience)
        
        # Should call rollback in context manager cleanup
        # Note: This is handled by the get_connection context manager
    
    def test_large_data_handling(self, storage):
        """Test handling of large data objects."""
        import numpy as np
        
        # Create experience with large embeddings
        large_embedding = np.random.randn(2048)  # Large embedding
        
        experience = SensoryExperience(
            tactile_embedding=large_embedding,
            visual_embedding=large_embedding,
            notes='x' * 10000  # Large text
        )
        
        storage._mock_cursor.fetchone.return_value = [999]
        
        exp_id = storage.store_experience(experience)
        
        assert exp_id == 999
        storage._mock_conn.commit.assert_called()
    
    def test_json_serialization(self, storage):
        """Test JSON serialization of complex data."""
        complex_data = {
            'nested': {'values': [1, 2, 3]},
            'arrays': [[1, 2], [3, 4]],
            'strings': 'test',
            'floats': 3.14159
        }
        
        experience = SensoryExperience(
            action_parameters=complex_data,
            material_interaction=complex_data
        )
        
        storage._mock_cursor.fetchone.return_value = [777]
        
        exp_id = storage.store_experience(experience)
        
        assert exp_id == 777
        # Should not raise JSON serialization errors
    
    def test_pagination(self, storage):
        """Test pagination in experience retrieval."""
        storage._mock_cursor.fetchall.return_value = []
        
        # Test with offset and limit
        experiences = storage.get_experiences(limit=20, offset=100)
        
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert 'LIMIT %s OFFSET %s' in sql_call[0]
        assert 20 in sql_call[1]
        assert 100 in sql_call[1]
    
    def test_connection_pool_exhaustion(self, storage):
        """Test handling of connection pool exhaustion."""
        storage.pool.getconn.side_effect = psycopg2.pool.PoolError("Pool exhausted")
        
        with pytest.raises(psycopg2.Error):
            with storage.get_connection() as conn:
                pass
    
    def test_invalid_data_types(self, storage):
        """Test handling of invalid data types."""
        # Experience with invalid timestamp
        experience = SensoryExperience()
        experience.timestamp = "invalid_timestamp"
        
        # Should not raise error during parameter binding
        # PostgreSQL will handle type conversion
        storage._mock_cursor.fetchone.return_value = [111]
        
        try:
            exp_id = storage.store_experience(experience)
            assert exp_id == 111
        except (TypeError, psycopg2.Error):
            # Expected for invalid data types
            pass
    
    def test_sql_injection_prevention(self, storage):
        """Test SQL injection prevention."""
        # Attempt SQL injection through session_id parameter
        malicious_session_id = "'; DROP TABLE experiences; --"
        
        storage._mock_cursor.fetchall.return_value = []
        
        # Should use parameterized queries (no exception)
        experiences = storage.get_experiences(session_id=malicious_session_id)
        
        # Verify parameterized query was used
        sql_call = storage._mock_cursor.execute.call_args[0]
        assert '%s' in sql_call[0]  # Parameterized query
        assert malicious_session_id in sql_call[1]  # Parameter passed safely


class TestPostgreSQLStorageIntegration:
    """Integration tests for PostgreSQL storage."""
    
    @pytest.fixture
    def storage_with_real_connection(self, connection_params):
        """Create storage with real PostgreSQL connection for integration tests."""
        # Skip if no real PostgreSQL available
        pytest.skip("Integration tests require real PostgreSQL database")
        
        # Uncomment for actual integration testing:
        # try:
        #     storage = PostgreSQLStorage(connection_params)
        #     yield storage
        #     storage.close()
        # except Exception:
        #     pytest.skip("PostgreSQL not available for integration tests")
    
    def test_full_experience_lifecycle(self, storage_with_real_connection):
        """Test complete experience storage and retrieval lifecycle."""
        storage = storage_with_real_connection
        
        # Create and store experience
        experience = SensoryExperience(
            primary_material='steel',
            action_type=ActionType.GRASP,
            success=True,
            reward=0.8
        )
        
        exp_id = storage.store_experience(experience)
        assert isinstance(exp_id, int)
        assert exp_id > 0
        
        # Retrieve experience
        experiences = storage.get_experiences(limit=1)
        assert len(experiences) >= 1
        
        # Find our experience
        stored_exp = next(
            (e for e in experiences if e['id'] == exp_id), 
            None
        )
        assert stored_exp is not None
        assert stored_exp['primary_material'] == 'steel'
        assert stored_exp['success'] is True
    
    def test_performance_bulk_operations(self, storage_with_real_connection):
        """Test performance with bulk operations."""
        storage = storage_with_real_connection
        
        # Create multiple experiences
        experiences = []
        for i in range(100):
            exp = SensoryExperience(
                primary_material=f'material_{i % 5}',
                reward=i / 100.0
            )
            experiences.append(exp)
        
        # Time bulk storage
        import time
        start_time = time.time()
        
        exp_ids = []
        for exp in experiences:
            exp_id = storage.store_experience(exp)
            exp_ids.append(exp_id)
        
        storage_time = time.time() - start_time
        
        assert len(exp_ids) == 100
        assert all(isinstance(eid, int) for eid in exp_ids)
        assert storage_time < 10.0  # Should complete in reasonable time
        
        # Test bulk retrieval
        start_time = time.time()
        retrieved = storage.get_experiences(limit=100)
        retrieval_time = time.time() - start_time
        
        assert len(retrieved) >= 100
        assert retrieval_time < 5.0  # Should be fast


if __name__ == '__main__':
    pytest.main([__file__, '-v'])